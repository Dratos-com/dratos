from dratos.engines.serve.engines.openai_engine import OpenAIEngine, OpenAIEngineConfig
from typing import Any, Dict, List, Union, Type, Optional
import pyarrow as pa
import mlflow
from openai import AsyncOpenAI, OpenAI


class VLLMEngine(OpenAIEngine):
    def __init__(
        self,
        config: OpenAIEngineConfig = OpenAIEngineConfig(),
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(config)
        self.api_key = api_key
        self.base_url = base_url
        self.client: Optional[AsyncOpenAI | OpenAI] = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    async def initialize(self) -> None:
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        mlflow.openai.autolog()

    async def generate_structured(
        self,
        prompt: Union[str, List[str]],
        structure: Union[str, Dict, pa.Schema],
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ) -> Any:
        # Implement structured generation logic here
        response_format = {"type": "json_object"}

        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format=response_format,
            **kwargs,
        )
        return response


if __name__ == "__main__":
    import asyncio
    from dratos.config import Config

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
