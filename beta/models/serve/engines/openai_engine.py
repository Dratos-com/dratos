import os
from typing import Any, Dict, List, Union, Optional
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, Field
from .base_engine import BaseEngine
from beta.models.serve.utils.prompt_utils import prompt
from beta.data.obj.base import DataObjectSchema
import mlflow
import json
import pyarrow as pa
from outlines.processors import (
    OutlinesLogitsProcessor,
    JSONLogitsProcessor,
    RegexLogitsProcessor,
)
from outlines.fsm.json_schema import build_regex_from_schema, convert_json_schema_to_str
from beta.data.obj.base import DataObject

class OpenAIEngineConfig(DataObject):
    """Configuration for OpenAI Engine."""

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
    top_k: int = Field(0, description="Sample from the top_k most likely tokens.")
    # Add other supported parameters here


class OpenAIEngine(BaseEngine):
    def __init__(
        self,
        model_name: str = "NousResearch/Meta-Llama-3-8B-Instruct",
        mlflow_client: mlflow.tracking.MlflowClient,
        config: OpenAIEngineConfig = OpenAIEngineConfig(),
    ):
        super().__init__(model_name, mlflow_client, config=config)
        self.client: Optional[Union[OpenAI | AsyncOpenAI]] = None
        self.config = config

    async def initialize(self) -> None:
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        mlflow.openai.autolog()

    async def generate(self, 
        prompt: Union[str, List[str]],
          **kwargs
          ) -> Any:
        
        if not self.client:
            await self.initialize()

        async with mlflow.start_run(run_name=f"OpenAI_{self.model_name}_generation"):

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": p}
                    for p in (prompt if isinstance(prompt, list) else [prompt])
                ],
                **{**self.config.dict(), **kwargs},
            )
            result = [choice.message.content for choice in response.choices]
            await self.log_metrics({"tokens": response.usage.total_tokens})
            return result[0] if isinstance(prompt, str) else result

    async def generate_structured(
        self,
        prompt: Union[str, List[str]],
        structure: Union[str, Dict, pa.Schema, DataObject, ],
        **kwargs,
    ) -> Any:
        if isinstance(structure, str):
            self.set_logits_processor(
                RegexLogitsProcessor(structure, self.client.tokenizer)
            )
        elif isinstance(structure, dict):
            schema_str = convert_json_schema_to_str(structure)
            regex_str = build_regex_from_schema(schema_str)
            self.set_logits_processor(
                RegexLogitsProcessor(regex_str, self.client.tokenizer)
            )
        elif isinstance(structure, pa.Schema):
            json_schema = self._arrow_to_json_schema(structure)
            schema_str = convert_json_schema_to_str(json_schema)
            regex_str = build_regex_from_schema(schema_str)
            self.set_logits_processor(
                RegexLogitsProcessor(regex_str, self.client.tokenizer)
            )
        elif issubclass(structure, DataObjectSchema):
            json_schema = structure.to_json_schema()
            schema_str = convert_json_schema_to_str(json_schema)
            regex_str = build_regex_from_schema(schema_str)
            self.set_logits_processor(
                RegexLogitsProcessor(regex_str, self.client.tokenizer)
            )
        else:
            raise ValueError("Unsupported structure type")

        result = await self.generate(prompt, **kwargs)

        if isinstance(structure, (dict, pa.Schema, Type[DataObject])):
            parsed_result = (
                json.loads(result)
                if isinstance(prompt, str)
                else [json.loads(r) for r in result]
            )
            if issubclass(structure, DataObject):
                return (
                    structure.from_json(json.dumps(parsed_result))
                    if isinstance(prompt, str)
                    else [structure.from_json(json.dumps(r)) for r in parsed_result]
                )
            return parsed_result
        return result

    @staticmethod
    def _arrow_to_json_schema(arrow_schema: pa.Schema) -> Dict[str, Any]:
        json_schema = {"type": "object", "properties": {}}
        for field in arrow_schema:
            json_schema["properties"][field.name] = {
                "type": OpenAIEngine._arrow_type_to_json_type(field.type)
            }
        return json_schema

    @staticmethod
    def _arrow_type_to_json_type(arrow_type: pa.DataType) -> str:
        if pa.types.is_string(arrow_type):
            return "string"
        elif pa.types.is_integer(arrow_type):
            return "integer"
        elif pa.types.is_floating(arrow_type):
            return "number"
        elif pa.types.is_boolean(arrow_type):
            return "boolean"
        elif pa.types.is_timestamp(arrow_type):
            return "string"  # Use ISO 8601 format for timestamps
        else:
            return "string"  # Default to string for complex types

    @property
    def supported_tasks(self) -> List[str]:
        return ["text-generation", "chat", "structured-generation"]

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "api_key": "***",  # Masked for security
            **self.config.dict(),
        }

    async def shutdown(self) -> None:
        if self.client:
            await self.client.close()

    async def log_metrics(self, metrics: Dict[str, Any]) -> None:
        for key, value in metrics.items():
            self.mlflow_client.log_metric(key, value)


if __name__ == "__main__":
    import asyncio
    from beta.config import Config
    from beta.models.serve.engines.openai_engine import OpenAIEngine

    async def main():
        config = Config()
        mlflow_client = config.get_mlflow()
        engine_config = OpenAIEngineConfig(temperature=0.7, max_tokens=100)
        engine = OpenAIEngine(
            model_name="gpt-4o", mlflow_client=mlflow_client, config=engine_config
        )
        await engine.initialize()

        result = await engine.generate(
            "Explain the concept of machine learning in simple terms.",
            task="text-generation",
        )
        print(result)

        await engine.shutdown()

    asyncio.run(main())
