"""Defines the Agent class and related types and functions."""

from __future__ import annotations
import asyncio
from typing import Protocol, List, Any, Optional, Union
from enum import Enum
from datetime import datetime
import logging
from altair import Vector2string
from daft import DataFrame, Schema, DataType, lit, col
import daft
import lancedb
from lancedb.embeddings import get_registry
import numpy as np
from pydantic import BaseModel, Field
import pyarrow as pa
from ray.serve.handle import DeploymentResponse
import uuid
import os

from ..data.obj.artifacts.artifact_obj import Artifact
from ..models.obj.base_language_model import LLM
from ..models.serve.engines.base_engine import BaseEngine
from ..models.serve.engines.openai_engine import OpenAIEngine, OpenAIEngineConfig
from ..data.obj.result import Result
from ..data.obj.memory import Memory, MemoryStore
from ..tools.git_api import GitAPI


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

    def execute(self, data: Any) -> Result[Any, Exception]: ...


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
            raise ValueError("Must be a Schema instance")
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
        model: LLM = LLM(
            "openai/gpt-4o", engine=OpenAIEngine(config=OpenAIEngineConfig())
        ),
        embedding: LLM = LLM(
            "openai/text-embedding-3-large",
            engine=OpenAIEngine(config=OpenAIEngineConfig()),
        ),
        stt: LLM = LLM(
            "openai/whisper-1", engine=OpenAIEngine(config=OpenAIEngineConfig())
        ),
        tools: Optional[List[ToolInterface]] = None,
        metadata: Optional[Metadata] = None,
        engine: Optional[BaseEngine] = None,
        artifacts: Optional[List[DataFrame]] = None,
        memory: Optional[List[DataFrame]] = None,
        memory_db_uri: Optional[str] = None,
        git_repo_path: Optional[str] = None,
        is_async: bool = False,
    ):
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

        self.lancedb_client = lancedb.connect(memory_db_uri)
        logging.info(f"Lancedb client connected: {self.lancedb_client}")

        # Initialize the embedding function
        self.embedding_func = (
            get_registry()
            .get("sentence-transformers")
            .create(name="BAAI/bge-small-en-v1.5")
        )
        self.memory_store = MemoryStore(memory_db_uri)
        self.git_api = GitAPI(git_repo_path)



    async def process(
        self,
        prompt: Optional[Prompt | str] = None,
        messages: Optional[List[Message]] = None,
        speech: Optional[
            Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]]
        ] = None,
    ) -> str:
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
                messages.append(
                    Message(role="system", content=f"Transcription: {transcription}")
                )

        logging.info("Sending request to model")
        if isinstance(prompt, Prompt):
            prompt = prompt.content

        logging.info(f"Prompt: {prompt}")

        if messages is None:
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ]

        if self.artifacts:
            for artifact in self.artifacts:
                # store in lancedb
                await self.store_in_lancedb(artifact)

        llm_response: DeploymentResponse = await self.model.generate(
            input=prompt, messages=messages
        )
        logging.info("Awaiting model response")
        logging.info(f"Model response received: {llm_response}")

        self.status = AgentStatus.IDLE
        logging.info(f"Agent status set to {self.status}")
        return llm_response

    async def store_in_lancedb(self, artifact: Artifact):
        vector_table_name = "artifacts_vectors"
        try:
            logging.info(f"Storing artifact in LanceDB: {artifact}")
            # Get the DataFrame from the Artifact
            artifact_df = artifact.df
            # gzip unzip the payload column
            import gzip
            
            def unzip_payload(zipped_payload: DataType.string):
                try:
                    return gzip.decompress(zipped_payload)

                except Exception as e:
                    logging.warning(f"Failed to unzip payload: {e}")
                    return zipped_payload  # Return original payload if unzipping fails

            unzip_payloads = unzip_payload(artifact_df.select("payload"))
            
            @daft.udf(
                return_dtype=DataType.embedding(
                    DataType.float32(), size=self.embedding_func.dimension
                )
            )
            def get_embeddings(payload: DataType.string):
                return self.embedding_func.compute_source_embeddings([payload])

            # Vectorize all payloads using the embedding function
            vectors_df = unzip_payloads.with_column(
                "vector", get_embeddings(unzip_payloads.select("payload"))
            )
            vectors_df.collect()
            # Check the dimensions of the vectors
            vector_dim = len(vectors_df.select("vector").to_pylist()[0])
            logging.info(
                f"Vector dimension: {vector_dim}, expected: {self.embedding_func.dimension}"
            )

            # Create a PyArrow table with the vectors
            pa_table = vectors_df.to_arrow()

            # Check if the vector table exists
            if vector_table_name not in self.lancedb_client.table_names():
                # Create a new vector table
                lancedb_table = self.lancedb_client.create_table(
                    vector_table_name, pa_table
                )
                logging.info(
                    f"Created new vector table '{vector_table_name}' in LanceDB."
                )
            else:
                # Append to the existing vector table
                lancedb_table = self.lancedb_client.open_table(vector_table_name)
                lancedb_table.add(pa_table)
                logging.info(
                    f"Appended to existing table '{vector_table_name}' in LanceDB."
                )

            return lancedb_table
        except Exception as e:
            logging.error(f"Failed to store artifact vectors in LanceDB: {e}")
            raise

    async def add_memory(self, content: str) -> str:
        vector = await self.model.engine.get_embedding(content)
        memory = Memory(id=str(uuid.uuid4()), content=content, vector=vector)
        self.memory_store.add_memory(memory)
        return self.git_api.commit_memory(f"Added memory: {memory.id}")

    async def search_memories(self, query: str, limit: int = 5) -> List[Memory]:
        query_vector = await self.model.engine.get_embedding(query)
        return self.memory_store.search_memories(query_vector, limit)

    def create_memory_branch(self, branch_name: str):
        self.git_api.create_branch(branch_name)

    def switch_memory_branch(self, branch_name: str):
        self.git_api.switch_branch(branch_name)

    def time_travel(self, commit_hash: str):
        self.git_api.checkout_commit(commit_hash)
        # Reload memories from the current state
        self._reload_memories()

    def _reload_memories(self):
        # This method should reload memories from the LanceDB table
        # based on the current Git state
        self.memory = self.memory_store.get_all_memories()

    async def execute_pipeline(self, input_data: Any) -> Result[Any, Exception]:
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

        # Add memory of the execution
        await self.add_memory(f"Executed pipeline with input: {input_data}")

        return result

    def get_status(self) -> AgentStatus:
        return self.status

    async def infer_action(
        self, prompt: Prompt, actions: List[str], tools: List[ToolInterface]
    ) -> str:
        """
        Infer the action to perform based on the prompt and tools.
        """

        action_prompt = f"{prompt.content}\n\nAvailable actions: {', '.join(actions)}\n\nBased on the prompt, which action should be taken?"

        action_response: DeploymentResponse = await self.model.generate.remote(
            prompt=Prompt(content=action_prompt)
        )
        inferred_action = await action_response

        # Clean up the response to match one of the available actions
        inferred_action = inferred_action.strip().lower()
        if inferred_action not in [action.lower() for action in actions]:
            # If the inferred action doesn't match any available action, default to the first action
            print(
                f"Inferred action '{inferred_action}' not found in available actions. Defaulting to '{actions[0]}'."
            )
            inferred_action = actions[0]
        else:
            # Find the original action with correct capitalization
            inferred_action = next(
                action for action in actions if action.lower() == inferred_action
            )

        print(f"Inferred action: {inferred_action}")

        # Add memory of the inferred action
        await self.add_memory(f"Inferred action: {inferred_action} for prompt: {prompt.content}")

        return inferred_action

    def choose_tool(self, task: str, tools: List[ToolInterface]) -> ToolInterface:
        """
        Choose the tool to use based on the tools.
        """
        # TODO: Implement tool selection logic
        return next((tool for tool in tools if tool.can_handle(task)), None)


__all__ = ["Agent", "AgentStatus", "ToolInterface", "Prompt", "Message", "Metadata"]
