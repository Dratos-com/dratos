"""
This module defines the base language model class and related classes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import typing

if typing.TYPE_CHECKING:
    pass
from typing import List
from ..serve.engines.base_engine import BaseEngine
from ..serve.engines.openai_engine import OpenAIEngine, OpenAIEngineConfig


# Define the base input class
class Input(ABC):
    """
    Base class for input objects.
    """
    @abstractmethod
    def to_text(self) -> str:
        pass


# Define a text input subclass
class TextInput(Input):
    """
    Input class for text data.
    """
    def __init__(self, text: str):
        self.text = text

    def to_text(self) -> str:
        return self.text


# Define an image input subclass (example)
class ImageInput(Input):
    """
    Input class for image data.
    """
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
        config: OpenAIEngineConfig = OpenAIEngineConfig(),
        engine: BaseEngine = OpenAIEngine(OpenAIEngineConfig())
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

    
