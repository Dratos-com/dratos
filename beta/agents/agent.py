""" Defines the Agent class and related types and functions. """
from __future__ import annotations
import asyncio
from typing import Protocol, List, Any, Optional, Union
from enum import Enum
from datetime import datetime
import logging
from altair import Vector2string
from daft import DataFrame, Schema, DataType, lit, col
import lancedb
import numpy as np
from pydantic import BaseModel, Field
from ray.serve.handle import DeploymentResponse

from ..data.obj.artifacts.artifact_obj import Artifact
from ..models.obj.base_language_model import LLM
from ..models.serve.engines.base_engine import BaseEngine
from ..models.serve.engines.openai_engine import OpenAIEngine, OpenAIEngineConfig
from ..data.obj.result import Result


class AgentStatus(str, Enum):
    """
    Enum representing the status of an agent.
    """
    INIT = "initializing"
    IDLE = "idle"
    PENDING = "pending"
    WAITING = "waiting"
    PROCESSING = "processing"


class ToolInterface(Protocol):
    """
    Protocol representing a tool that can be executed by an agent.
    """
    def execute(self, data: Any) -> Result[Any, Exception]:
        ...


class Prompt(BaseModel):
    """
    Pydantic model representing a prompt.
    """
    content: str


class Message(BaseModel):
    """
    Pydantic model representing a message.
    """
    role: str
    content: str


class SchemaWrapper(BaseModel):
    """
    Custom type for the schema field.
    """
    schema: Any

    class Config:
        """
        Pydantic configuration for the SchemaWrapper class.
        """
        arbitrary_types_allowed = True

    @classmethod
    def __get_validators__(cls):
        """
        Get validators for the SchemaWrapper class.
        """
        yield cls.validate

    @classmethod
    def validate(cls, value):
        """
        Validate the schema.
        """
        if not isinstance(value, Schema):
            raise ValueError('Must be a Schema instance')
        return cls(schema=value)


class Metadata(BaseModel):
    """
    Pydantic model representing metadata for an agent.
    """
    schema: SchemaWrapper
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance: Optional[str] = None

    def serialize(self, serializer):
        """
        Serialize the metadata.
        """
        # Implement serialization logic if needed


class Agent:
    """
    Orchestrates the execution of models and tools.

    Attributes:
        name (str): The name of the agent.
        model (OpenAIEngine): Instance of the model engine.
        embedding (OpenAIEngine): Instance of the embedding engine.
        stt (OpenAIEngine): Instance of the speech-to-text engine.
        tools (List[ToolInterface]): List of tools the agent can utilize.
        metadata (Metadata): Metadata information for the agent.
        engine (BaseEngine): Inference engine.
        artifacts (List[DataFrame]): List of artifacts.
        memory (List[DataFrame]): List of memory.
        is_async (bool): Use asynchrony (i.e. for streaming).
    """
    def __init__(
            self,
            name: str,
            model: LLM = LLM("openai/gpt-4o", engine=OpenAIEngine(config=OpenAIEngineConfig())),
            embedding: LLM = LLM("openai/text-embedding-3-large", engine=OpenAIEngine(config=OpenAIEngineConfig())),
            stt: LLM = LLM("openai/whisper-1", engine=OpenAIEngine(config=OpenAIEngineConfig())),
            tools: Optional[List[ToolInterface]] = None,
            metadata: Optional[Metadata] = None,
            engine: Optional[BaseEngine] = None,
            artifacts: Optional[List[DataFrame]] = None,
            memory: Optional[List[DataFrame]] = None,
            is_async: bool = False):
        self.name = name
        self.model = model
        self.embedding = embedding
        self.stt = stt
        self.tools = tools or []
        self.metadata = metadata
        self.engine = engine
        self.is_async = is_async
        self.status = AgentStatus.INIT
        self.memory = memory
        self.artifacts = artifacts

        self.lancedb_client = lancedb.connect("~/.lancedb")
        logging.info(f"Lancedb client connected: {self.lancedb_client}")

    async def process(self, prompt: Optional[Prompt | str] = None,
                      messages: Optional[List[Message]] = None,
                      speech: Optional[Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]] = None) -> str:
        logging.info(f"Agent {self.name} called with prompt: {prompt}")
        self.status = AgentStatus.PROCESSING
        logging.info(f"Agent status set to {self.status}")

        if self.memory:
            for df in self.memory:
                logging.info(f"Memory: {df}")
                messages.append(Message(role="user", content=f"Memory: {df}"))

        if speech is not None:
            logging.info("Processing speech input")
            stt_response: DeploymentResponse = await self.stt.generate.remote(speech)
            transcription = await stt_response
            logging.info(f"Speech transcribed: {transcription}")
            if prompt:
                prompt.content += f"\nTranscription: {transcription}"
            elif messages:
                messages.append(Message(role="system", content=f"Transcription: {transcription}"))

        logging.info("Sending request to model")
        if isinstance(prompt, Prompt):
            prompt = prompt.content

        logging.info(f"Prompt: {prompt}")
        
        if messages is None:
            messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]
        
        if self.artifacts:
            for artifact in self.artifacts:
                # store in lancedb
                await self.store_in_lancedb(artifact)

        llm_response: DeploymentResponse = await self.model.generate(prompt=prompt, messages=messages)
        logging.info("Awaiting model response")
        logging.info(f"Model response received: {llm_response}")

        self.status = AgentStatus.IDLE
        logging.info(f"Agent status set to {self.status}")
        return llm_response

    async def store_in_lancedb(self, artifact: Artifact):
        """
        Stores the vectors derived from the Artifact into LanceDB's vector storage.

        Args:
            artifact (Artifact): The Artifact object to process and store vectors from.
        """
        vector_table_name = "artifacts_vectors"
        try:
            logging.info(f"Storing artifact in LanceDB: {artifact}")

            # Get the DataFrame from the Artifact
            artifact_df = artifact.df

            async def vectorize_payload(payload):
                return await self.model.get_embedding(payload)

            # Collect all payloads
            payloads = artifact_df.select("payload").to_pandas()["payload"].tolist()

            # Vectorize all payloads asynchronously
            vectors = await asyncio.gather(*[vectorize_payload(payload) for payload in payloads])

            # Add the vectors to the DataFrame
            artifact_df = artifact_df.with_column("vector", vectors)

            # Check if the vector table exists
            if vector_table_name not in self.lancedb_client.table_names():
                # Create a new vector table with the 'vector' column
                self.lancedb_client.create_table(vector_table_name, artifact_df)
                logging.info(f"Created new vector table '{vector_table_name}' in LanceDB.")
            else:
                # Append vectors to the existing vector table
                existing_vector_table = self.lancedb_client.open_table(vector_table_name)
                existing_vector_table.add(artifact_df)
                logging.info(f"Appended vectors to existing table '{vector_table_name}' in LanceDB.")
        except Exception as e:
            logging.error(f"Failed to store artifact vectors in LanceDB: {e}")
            raise
    
    def execute_pipeline(self, input_data: Any) -> Result[Any, Exception]:
        """
        Executes the agent's processing pipeline.

        Args:
            input_data (Any): The initial input data for the pipeline.

        Returns:
            Result[Any, Exception]: The final output wrapped in a Result monad.
        """
        print(f"Agent {self.name} is executing pipeline.")
        result = Result.Ok(input_data)

        for tool in self.tools:
            result = result.bind(tool.execute)
            if result.is_error:
                print(f"Pipeline halted due to error: {result.value}")
                break

        if self.engine and not result.is_error:
            try:
                inference_result = self.engine.generate(result.value)
                result = Result.Ok(inference_result)
            except Exception as e:
                result = Result.Error(e)

        return result

    def get_status(self) -> AgentStatus:
        return self.status

    async def infer_action(self, prompt: Prompt, actions: List[str], tools: List[ToolInterface]) -> str:
        """
        Infer the action to perform based on the prompt and tools.
        """

        action_prompt = f"{prompt.content}\n\nAvailable actions: {', '.join(actions)}\n\nBased on the prompt, which action should be taken?"
        
        action_response: DeploymentResponse = await self.model.generate.remote(prompt=Prompt(content=action_prompt))
        inferred_action = await action_response

        # Clean up the response to match one of the available actions
        inferred_action = inferred_action.strip().lower()        
        if inferred_action not in [action.lower() for action in actions]:
            # If the inferred action doesn't match any available action, default to the first action
            print(f"Inferred action '{inferred_action}' not found in available actions. Defaulting to '{actions[0]}'.")
            inferred_action = actions[0]
        else:
            # Find the original action with correct capitalization
            inferred_action = next(action for action in actions if action.lower() == inferred_action)

        print(f"Inferred action: {inferred_action}")
        return inferred_action

    def choose_tool(self, task: str, tools: List[ToolInterface]) -> ToolInterface:
        """
        Choose the tool to use based on the tools.
        """
        # TODO: Implement tool selection logic
        return next((tool for tool in tools if tool.can_handle(task)), None)


__all__ = ["Agent", "AgentStatus", "ToolInterface", "Prompt", "Message", "Metadata"]
