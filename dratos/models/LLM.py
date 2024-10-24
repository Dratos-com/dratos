"""
This module defines the base language model class and related classes.
"""

from typing import Dict, List, AsyncIterator, Any
from dratos.models.engines.base_engine import BaseEngine

class LLM():
    def __init__(
        self,
        model_name: str,
        engine: BaseEngine
    ):
        self.model_name = model_name
        self.engine = engine

        self.support_tools = engine.support_tools(model_name)
        self.support_structured_output = engine.support_structured_output(model_name)

    async def initialize(self):
        self.engine.initialize()

    async def shutdown(self):
        self.engine.shutdown()

    async def sync_gen(self, 
                       response_model: str | Dict | None = None,
                       tools: List[Dict] = None,
                       messages: List[Dict[str, Any]] = None,
                       **kwargs
                       ) -> str:
        if tools is not None and response_model is not None:
            raise ValueError("Cannot use both 'tools' and 'response_model' simultaneously.")
        if not self.is_initialized:
            await self.initialize()
        return await self.engine.sync_gen(self.model_name, response_model, tools, messages, **kwargs)

    async def async_gen(self, 
                     messages: List[Dict[str, Any]] = None,
                     **kwargs
                     ) -> AsyncIterator[str]:
        if not self.is_initialized:
            await self.initialize()
        async for chunk in self.engine.async_gen(self.model_name, messages, **kwargs):
            yield chunk

    @property
    def supported_tasks(self) -> List[str]:
        return ["text-generation", "chat", "structured-generation"]

    @property
    def is_initialized(self) -> bool:
        return self.engine is not None

    
