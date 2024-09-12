from beta.models.serve.engines.openai_engine import OpenAIEngine, OpenAIEngineConfig
from typing import Any, Dict, List, Union, Type, Optional
import pyarrow as pa
import mlflow
from beta.data.obj.base import DataObject
import os
import vllm
from openai import AsyncOpenAI, OpenAI


class VLLMEngineConfig(OpenAIEngineConfig):
    """Configuration for VLLM Engine."""


class VLLMEngine(OpenAIEngine):
    def __init__(
        self,
        model_name: str,
        mlflow_client: mlflow.tracking.MlflowClient,
        config: VLLMEngineConfig = VLLMEngineConfig(),
    ):
        super().__init__(model_name, mlflow_client, config)
        self.client: Optional[AsyncOpenAI | OpenAI] = None

    async def initialize(self) -> None:
        api_key = os.getenv("VLLM_API_KEY")
        base_url = os.getenv("VLLM_BASE_URL")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        mlflow.openai.autolog()


if __name__ == "__main__":
    import asyncio
    from beta.config import Config

    async def main():
        config = Config()
        mlflow_client = config.get_mlflow()
        engine_config = OpenAIEngineConfig(temperature=0.7, max_tokens=100)
        engine = VLLMEngine(
            model_name="vllm-model", mlflow_client=mlflow_client, config=engine_config
        )
        await engine.initialize()

        result = await engine.generate(
            "Explain the concept of machine learning in simple terms.",
            task="text-generation",
        )
        print(result)

        await engine.shutdown()

    asyncio.run(main())