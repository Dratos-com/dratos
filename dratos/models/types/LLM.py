"""
This module defines the base language model class and related classes.
"""
from __future__ import annotations
from abc import abstractmethod

from typing import Dict, List, AsyncIterator
from dratos.models.providers.base_engine import BaseEngine



class LLM():
    def __init__(
        self,
        model_name: str,
        engine: BaseEngine
    ):
        self.model_name = model_name
        self.engine = engine
        if self.model_name.startswith("gpt-4o-"):
            self.support_pydantic = True
        else:
            self.support_pydantic = False

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

    
