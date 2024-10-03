"""
This module defines the base language model class and related classes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import typing

if typing.TYPE_CHECKING:
    pass
from typing import Dict, List, Optional, AsyncIterator
from dratos.engines.adapters.base_engine import BaseEngine
from dratos.engines.adapters.openai_engine import OpenAIEngine


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
    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.is_initialized:
            await self.initialize()
        return await self.client.generate.remote(prompt, **kwargs)

    @property
    def supported_tasks(self) -> List[str]:
        return ["text-generation", "chat", "structured-generation"]

    @property
    def is_initialized(self) -> bool:
        return self.engine is not None


class LLM(BaseLanguageModel):
    def __init__(
        self,
        model_name: str,
        engine: BaseEngine
    ):
        super().__init__(model_name, engine)
        self.model_name = model_name
        self.engine = engine

    async def initialize(self):
        self.engine.initialize()

    async def shutdown(self):
        pass

    @abstractmethod
    async def generate(self, 
                       prompt: dict, 
                       response_format: str | Dict | None = None,
                       tools: List[Dict] = None,
                       messages: List[Dict[str, str]] = None,
                       **kwargs
                       ) -> str:
        if tools is not None and response_format is not None:
            raise ValueError("Cannot use both 'tools' and 'output_structure' simultaneously.")
        if not self.is_initialized:
            await self.initialize()
        return await self.engine.generate(prompt, self.model_name, response_format, tools, messages, **kwargs)

    async def stream(self, 
                     prompt: dict, 
                     response_format: str | Dict | None = None,
                     tools: List[Dict] = None,
                     messages: List[Dict[str, str]] = None,
                     **kwargs
                     ) -> AsyncIterator[str]:
        if tools is not None and response_format is not None:
            raise ValueError("Cannot use both 'tools' and 'output_structure' simultaneously.")
        if not self.is_initialized:
            await self.initialize()
        async for chunk in self.engine.stream(prompt, self.model_name, response_format, tools, messages, **kwargs):
            yield chunk

    @property
    def supported_tasks(self) -> List[str]:
        return ["text-generation", "chat", "structured-generation"]

    @property
    def is_initialized(self) -> bool:
        return self.engine is not None

    
