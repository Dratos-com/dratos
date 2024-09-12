import os
from typing import Any, Dict, List, Union, Optional
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field

from beta.data.obj.artifacts.artifact_obj import Artifact
from .base_engine import BaseEngine
import mlflow
import json
import pyarrow as pa
from beta.data.obj import DataObject
import tiktoken

class OpenAIEngineConfig(DataObject):
    """Configuration for OpenAI Engine."""

    model_name: str = Field(
        "o1-preview",
        description="The name of the OpenAI model to use.",
    )
    stream: bool = Field(
        False,
        description="Whether to stream the response.",
    )
    temperature: float = Field(
        0.7,
        description="Sampling temperature, higher values means the model will take more risks.",
    )
    max_tokens: int = Field(
        100, description="The maximum number of tokens to generate in the completion."
    )
    top_p: float = Field(
        1.0,
        description="Nucleus sampling, where p is the probability of the top_p most likely tokens.",
    )
    top_k: int = Field(
        40,
        description="The number of top most likely tokens to consider for generation.",
    )
    frequency_penalty: float = Field(
        0.0,
        description="Penalize new tokens based on their existing frequency in the text so far.",
    )
    presence_penalty: float = Field(
        0.0,
        description="Penalize new tokens based on whether they appear in the text so far.",
    )
    n: int = Field(
        1,
        description="How many chat completion choices to generate for each input message.",
    )

from outlines.processors import (
            BaseLogitsProcessor,
            JSONLogitsProcessor,
            RegexLogitsProcessor,
        )
from outlines.fsm.json_schema import build_regex_from_schema

class OpenAIEngine(BaseEngine):
    def __init__(
        self,
        model_name: str = "o1-preview",
        mlflow_client: mlflow.tracking.MlflowClient,
        config: OpenAIEngineConfig = OpenAIEngineConfig(),
    ):
        super().__init__(model_name, mlflow_client, config=config)
        self.client: Optional[Union[OpenAI | AsyncOpenAI]] = None
        self.config = config
        self.logits_processor: Optional[BaseLogitsProcessor] = None

    async def initialize(self) -> None:
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        mlflow.openai.autolog()
    
    async def shutdown(self) -> None:
        if self.client:
            await self.client.close()

    async def get_model_info(self):
        model = self.client.models.retrieve(self.model_name)
        return model
    
    async def get_supported_models(self):
        if not self.client:
            await self.initialize()
        models = await self.client.models.list()
        return [model.id for model in models.data]

    async def generate(self,
        prompt: str,
        messages: List[Dict[str, str]], 
        response_format: BaseModel = None,
        **kwargs,
        ):
        
        if not self.client:
            await self.initialize()

        # Add the user prompt to the messages
        messages.append({"role": "user", "content": prompt})

        with self.mlflow_client.start_run(run_name=f"OpenAI_{self.model_name}_generation"):
            response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format=response_format,
                    **kwargs
                )
        result = response.choices[0].message.content
        self.log_metrics({"total_tokens": response.usage.total_tokens})
        self.log_metrics({"prompt_tokens": response.usage.prompt_tokens})
        self.log_metrics({"completion_tokens": response.usage.completion_tokens})
        self.log_metrics({"prompt_per_token_cost": response.usage.prompt_tokens * 0.00001})
        self.log_metrics({"completion_per_token_cost": response.usage.completion_tokens * 0.00003})
    
        return result 

    async def generate_structured(
        self,
        prompt: Union[str, List[str]],
        structure: Union[str, Dict, pa.Schema, BaseModel],
        logits_processor: Optional[BaseLogitsProcessor] = None,
        **kwargs,
    ):
        if logits_processor is None:
            logits_processor = self.get_logits_processor(structure)
        else:
            self.logits_processor(logits_processor)
        result = await self.generate(prompt, **kwargs)

        return result
    
    async def get_logits_processor(self, structure: Union[str, dict, pa.Schema, BaseModel]):
        


        if isinstance(structure, str):
            self.set_logits_processor(
                RegexLogitsProcessor(structure)
            )
        elif isinstance(structure, dict):
            regex_str = build_regex_from_schema(structure)
            self.set_logits_processor(
                RegexLogitsProcessor(regex_str, self.client.tokenizer)
            )
        elif isinstance(structure, pa.Schema):
            json_schema = structure
            schema_str = convert_json_schema_to_str(json_schema)
            regex_str = build_regex_from_schema(schema_str)
            self.set_logits_processor(
                RegexLogitsProcessor(regex_str, self.client.tokenizer)
            )
        elif issubclass(structure, BaseModel):
            json_schema = structure.model_json_schema()
            schema_str = str(json_schema)
            regex_str = build_regex_from_schema(schema_str)
            self.logits_processor = RegexLogitsProcessor(regex_str, self.client.models)

        else:
            raise ValueError("Unsupported structure type")
        
        return self.logits_processor


    @property
    def supported_tasks(self) -> List[str]:
        return ["text-generation", "chat", "structured-generation"]

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "api_key": "***",  # Masked for security
            **self.config.dict(),
        }



    #async def log_artifacts(self, artifacts: List[Artifact]) -> None:
    #    for artifact in artifacts:
    #        self.mlflow_client.log_artifact(run_id=self.mlflow_client.active_run().info.run_id, local_path=artifact.path)

