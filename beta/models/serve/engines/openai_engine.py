import os
from typing import Any, Coroutine, Dict, List, Union, Optional
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
    def __init__(self, data: Dict[str, Any] = None):
        super().__init__()
        if data is None:
            data = {}
        
        # Convert single values to lists
        self._data = {k: [v] if not isinstance(v, (list, tuple)) else v for k, v in data.items()}
        self.df = daft.from_pydict(self._data)

    def update(self, new_config: Dict[str, Any]) -> None:
        new_data = {k: [v] if not isinstance(v, (list, tuple)) else v for k, v in new_config.items()}
        self._data.update(new_data)
        self.df = daft.from_pydict(self._data)

    def get(self, name, default=None):
        if name in self._data:
            return self._data[name][0]
        return default

    def __getattr__(self, name):
        return self.get(name)


class OpenAIEngine(BaseEngine):
    def __init__(
        self,
        config: dict = OpenAIEngineConfig()
    ):
        super().__init__(config=config)
        self.client: Optional[Union[OpenAI | AsyncOpenAI]] = None

    def set_logits_processor(self, processor: OutlinesLogitsProcessor) -> None:
        self.logits_processor = processor

    def get_logits_processor(self) -> OutlinesLogitsProcessor:
        return self.logits_processor
    
    async def generate_structured(self, prompt: str | List[str], structure: str | Dict | Any, config: Optional[OpenAIEngineConfig]=None, **kwargs) -> Coroutine[Any, Any, Any]:
        if not self.client:
            await self.initialize(config if config else self.config)
        
        if config: 
            self.config = config
            self.model_name = config
            

        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        # Convert structure to a string representation
        if isinstance(structure, dict):
            structure_str = json.dumps(structure)
        elif isinstance(structure, pa.Schema):
            structure_str = structure.to_string()
        else:
            structure_str = str(structure)

        # Add the structure requirement to the prompt
        json_prompt = f"{prompt}\nPlease provide the answer in the following JSON structure: {structure_str}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides answers in the specified JSON format."},
            {"role": "user", "content": json_prompt}
        ]

        response_format = {"type": "json_object"}

        response = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format=response_format,
            **kwargs
        )

        result = response.choices[0].message.content
        return result

    async def initialize(self, config: OpenAIEngineConfig) -> None:
        self.config = config
        self.client = AsyncOpenAI(
            api_key=self.config.get("api_key"),
            base_url=self.config.get("base_url", "https://api.openai.com/v1")
        )

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
        messages: List[Dict[str, str]] = [{"role": "system", "content": "You are a helpful assistant"}],
        response_format: BaseModel = None,
        **kwargs,
    ):
        if not self.client:
            await self.initialize()

        # Add the user prompt to the messages
        if messages is not None:
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]

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
