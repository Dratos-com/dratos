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
from beta.models.serve.engines import BaseEngine, OpenAIEngine, OpenAIEngineConfig


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


class BaseLanguageModel:
    """Base class for language models."""

    def __init__(
        self,
        model_name: str = None,
        engine: BaseEngine = OpenAIEngine(),
    ):
        self.model_name = model_name
        self.engine = engine
        self._is_initialized = False

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def shutdown(self):
        pass

    @abstractmethod
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

    @property
    def is_initialized(self) -> bool:
        return self.engine is not None


class LLM(BaseLanguageModel):
    def __init__(
        self,
        model_name: str = None,
        engine: BaseEngine = OpenAIEngine(),
        config=OpenAIEngineConfig,
    ):
        super().__init__(model_name, engine)
        self.config = config

    async def initialize(self):
        self.engine.initialize(self.config)

    async def shutdown(self):
        pass

    @abstractmethod
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

    @property
    def is_initialized(self) -> bool:
        return self.engine is not None

    
