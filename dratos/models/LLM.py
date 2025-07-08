"""
This module defines the base language model class and related classes.
"""

from typing import Dict, List, AsyncIterator, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
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

    def initialize(self, asynchronous: bool = False):
        self.engine.initialize(asynchronous=asynchronous)

    def shutdown(self):
        self.engine.shutdown()

    def sync_gen(self, 
                response_model: str | Dict | None = None,
                tools: List[Dict] = None,
                messages: List[Dict[str, Any]] = None,
                timeout: float = 300.0,
                **kwargs
                ) -> str:
        if tools is not None and response_model is not None:
            raise ValueError("Cannot use both 'tools' and 'response_model' simultaneously.")
        
        # Always reinitialize the engine before each call
        self.initialize(asynchronous=False)
        
        def call_engine():
            return self.engine.sync_gen(self.model_name, response_model, tools, messages, **kwargs)
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(call_engine)
                return future.result(timeout=timeout)
        except TimeoutError:
            raise TimeoutError(f"LLM response timed out after {timeout} seconds")
        finally:
            # Ensure the engine is properly shut down after each call
            self.shutdown()

    async def async_gen(self, 
                     response_model: str | Dict | None = None,
                     tools: List[Dict] = None,
                     messages: List[Dict[str, Any]] = None,
                     timeout: float = 30.0,
                     **kwargs
                     ) -> AsyncIterator[str]:
        if tools is not None and response_model is not None:
            raise ValueError("Cannot use both 'tools' and 'response_model' simultaneously.")
        
        # Always reinitialize the engine before each call
        self.initialize(asynchronous=True)
        
        try:
            async def process_stream():
                async for chunk in self.engine.async_gen(self.model_name, response_model, tools, messages, **kwargs):
                    yield chunk
            
            # Create the stream processor
            stream_processor = process_stream()
            # Wrap it in wait_for and await the result
            async with asyncio.timeout(timeout):
                async for chunk in stream_processor:
                    yield chunk
        except asyncio.TimeoutError:
            raise TimeoutError(f"LLM response timed out after {timeout} seconds")
        finally:
            # Ensure the engine is properly shut down after each call
            self.shutdown()

    @property
    def supported_tasks(self) -> List[str]:
        return ["text-generation", "chat", "structured-generation"]

    @property
    def is_initialized(self) -> bool:
        return self.engine is not None

    
