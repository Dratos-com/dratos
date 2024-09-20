import os
from typing import Any, Dict, List, Union, Optional
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field

from beta.data.obj.artifacts.artifact_obj import Artifact
from .base_engine import BaseEngine, BaseEngineConfig
import mlflow
import json
import pyarrow as pa
from beta.data.obj import DataObject
import tiktoken
from outlines.processors.base_logits_processor import OutlinesLogitsProcessor
from outlines.processors.structured import JSONLogitsProcessor
from outlines.processors.structured import RegexLogitsProcessor
from outlines.fsm.json_schema import build_regex_from_schema
import lark
import daft

class OpenAIEngineConfig(BaseEngineConfig):
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

    def __init__(self, data: Dict[str, Any] = None):
        super().__init__()
        if data is None:
            data = {}
        
        # Convert single values to lists
        data = {k: [v] if not isinstance(v, list) else v for k, v in data.items()}
        
        self.df = daft.from_pydict(data)

    def update(self, new_config: Dict[str, Any]) -> None:
        new_data = {k: [v] if not isinstance(v, list) else v for k, v in new_config.items()}
        new_df = daft.from_pydict(new_data)
        self.df = self.df.update(new_df)

    def __getattr__(self, name):
        if name in self.df.columns:
            return self.df[name][0]

    def __call__(self):
        return self.model_dump()


class OpenAIEngine(BaseEngine):
    def __init__(
        self,
        model_name: str = "openai/o1-preview",
        config: dict = OpenAIEngineConfig(),
        mlflow_client: mlflow.tracking.MlflowClient = None,
    ):
        super().__init__(model_name=model_name, mlflow_client=mlflow_client, config=config)
        self.client: Optional[Union[OpenAI | AsyncOpenAI]] = None
        self.config = config
        self.logits_processor: Optional[OutlinesLogitsProcessor] = None
        self.model_name = model_name

    async def initialize(self) -> None:
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    async def generate(
        self,
        prompt: str,
        messages: List[Dict[str, str]],
        response_format: BaseModel = None,
        **kwargs,
    ):
        if not self.client:
            await self.initialize()

        # Add the user prompt to the messages
        messages.append({"role": "user", "content": prompt})
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format=response_format,
            **kwargs,
        )
        result = response.choices[0].message.content

        return result

    async def generate_data(
        self,
        prompt: Union[str, List[str]],
        structure: Union[str, Dict, pa.Schema, DataObject],
        grammar: lark.Lark = None,
        logits_processor: Optional[OutlinesLogitsProcessor] = None,
        **kwargs,
    ):
        if logits_processor is None:
            logits_processor = self.get_logits_processor(structure)
        else:
            self.logits_processor(logits_processor)
        result = await self.generate(prompt, **kwargs)

        return result

    async def get_logits_processor(
        self, structure: Union[str, dict, pa.Schema, BaseModel]
    ):
        """Get the logits processor for the given structure."""
        if isinstance(structure, str):
            self.set_logits_processor(RegexLogitsProcessor(structure))
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

    async def log_artifacts(self, artifacts: List[Artifact]) -> None:
        for artifact in artifacts:
            if self.mlflow_client:
                self.mlflow_client.log_artifact(
                    run_id=self.mlflow_client.active_run().info.run_id,
                    local_path=artifact.path,
                )
