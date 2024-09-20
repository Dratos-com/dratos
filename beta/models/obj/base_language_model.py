from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
import os
from typing import List, Optional
from outlines import models, generate
import ray
from ray.actor import ActorHandle
import mlflow  # Add this import
from abc import ABC, abstractmethod


# Define the base input class
class Input(ABC):
    @abstractmethod
    def to_text(self) -> str:
        pass


# Define a text input subclass
class TextInput(Input):
    def __init__(self, text: str):
        self.text = text

    def to_text(self) -> str:
        return self.text


# Define an image input subclass (example)
class ImageInput(Input):
    def __init__(self, image_path: str):
        self.image_path = image_path

    def to_text(self) -> str:
        # Placeholder for image to text conversion logic
        return "Converted text from image"


class BaseLanguageModel(models.VLLM):
    def __init__(
        self,
        model_name: str,
        mlflow_client: mlflow.tracking.MlflowClient,
        config: dict = None,
    ):
        super().__init__(model_name, mlflow_client, config=config)
        self.client: Optional[ActorHandle] = None
        self._is_initialized = False

    async def initialize(self):
        raise NotImplementedError

    async def shutdown(self):
        if self.client:
            await self.client.shutdown.remote()
        self._is_initialized = False

    async def generate(self, input: Input, **kwargs) -> str:
        if not self.is_initialized:
            await self.initialize()
        prompt = input.to_text()
        return await self.client.generate.remote(prompt, **kwargs)

    async def generate_structured(
        self, input: Input, structure: dict, **kwargs
    ) -> dict:
        if not self.is_initialized:
            await self.initialize()
        prompt = input.to_text()
        return await self.client.generate_structured.remote(prompt, structure, **kwargs)

    @property
    def supported_tasks(self) -> List[str]:
        return ["text-generation", "chat", "structured-generation"]

    def get_model_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "api_key": "***",  # Masked for security
            **self.config,
        }

    @property
    def is_initialized(self) -> bool:
        return self.client is not None

    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        if not self.is_initialized:
            await self.initialize()
        return await self.client.batch_generate.remote(prompts, **kwargs)

    async def batch_generate_structured(
        self, prompts: List[str], structure: dict, **kwargs
    ) -> List[dict]:
        if not self.is_initialized:
            await self.initialize()
        return await self.client.batch_generate_structured.remote(
            prompts, structure, **kwargs
        )

    async def get_token_count(self, text: str) -> int:
        if not self.is_initialized:
            await self.initialize()
        return await self.client.get_token_count.remote(text)

    async def get_model_info(self) -> dict:
        if not self.is_initialized:
            await self.initialize()
        return await self.client.get_model_info.remote()

    @property
    def max_tokens(self) -> int:
        return self.config.get("max_tokens", 2048)  # Default to 2048 if not specified

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

    @classmethod
    async def register_model(
        cls,
        model_name: str,
        model_path: str,
        mlflow_client: mlflow.tracking.MlflowClient,
    ):
        # Register the model with MLflow
        model_uri = f"models:/{model_name}/latest"
        mlflow.register_model(model_uri, model_name)

        # Log model details
        with mlflow.start_run():
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_path", model_path)
            mlflow.log_artifact(model_path)

        print(f"Model {model_name} registered successfully.")


class LLM(BaseLanguageModel):
    pass
