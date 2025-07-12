"""
This module defines the base language model class and related classes.
"""

from typing import Dict, List, AsyncIterator, Any
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dratos.models.engines.base_engine import BaseEngine

import logging
logger = logging.getLogger(__name__)

# Compatibility for Python < 3.10
try:
    from builtins import aiter, anext
except ImportError:
    # Fallback for older Python versions
    def aiter(async_iterable):
        return async_iterable.__aiter__()
    
    async def anext(async_iterator):
        return await async_iterator.__anext__()

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
                     start_timeout: float = 60.0,    # Timeout for first token
                     token_gap_timeout: float = 30.0, # Timeout between tokens
                     **kwargs
                     ) -> AsyncIterator[str]:
        if tools is not None and response_model is not None:
            raise ValueError("Cannot use both 'tools' and 'response_model' simultaneously.")
        
        stream_id = f"{self.model_name}_{int(time.time()*1000)}"
        logger.info(f"🔄 LLM.async_gen starting - Stream ID: {stream_id}")
        
        # Always reinitialize the engine before each call
        self.initialize(asynchronous=True)
        
        tokens_yielded = 0
        stream_start_time = time.time()
        
        try:
            # Create the stream processor
            stream_processor = self.engine.async_gen(self.model_name, response_model, tools, messages, **kwargs)
            stream_iterator = aiter(stream_processor)
            
            # Wait for first token with start timeout
            try:
                first_chunk = await asyncio.wait_for(anext(stream_iterator), timeout=start_timeout)
                tokens_yielded += 1
                first_token_delay = time.time() - stream_start_time
                logger.info(f"🌊 First token received for {stream_id} after {first_token_delay:.3f}s")
                yield first_chunk
            except asyncio.TimeoutError:
                elapsed = time.time() - stream_start_time
                logger.error(f"⏰ Start timeout after {elapsed:.3f}s for {stream_id}")
                raise TimeoutError(f"LLM failed to start generating content within {start_timeout} seconds")
            except StopAsyncIteration:
                logger.info(f"✅ LLM stream completed immediately for {stream_id}")
                return
            
            # Continue with remaining tokens using token gap timeout
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(stream_iterator), timeout=token_gap_timeout)
                    tokens_yielded += 1
                    yield chunk
                except asyncio.TimeoutError:
                    logger.error(f"⏰ Token gap timeout after {token_gap_timeout}s for {stream_id}")
                    raise TimeoutError(f"LLM stopped generating content - {token_gap_timeout}s gap between tokens")
                except StopAsyncIteration:
                    break
                        
            logger.info(f"✅ LLM.async_gen completed normally for {stream_id} - {tokens_yielded} tokens")
                        
        except GeneratorExit:
            elapsed = time.time() - stream_start_time
            logger.warning(f"🛑 LLM GeneratorExit for {stream_id} after {elapsed:.3f}s, {tokens_yielded} tokens")
            raise
        except Exception as e:
            elapsed = time.time() - stream_start_time
            logger.error(f"💥 LLM exception for {stream_id} after {elapsed:.3f}s: {str(e)}")
            raise
        finally:
            # Ensure the engine is properly shut down after each call
            try:
                self.shutdown()
            except Exception as e:
                logger.error(f"❌ Engine shutdown failed for {stream_id}: {str(e)}")
            
            total_duration = time.time() - stream_start_time
            logger.info(f"🏁 LLM.async_gen finished for {stream_id} - {total_duration:.3f}s, {tokens_yielded} tokens")

    @property
    def supported_tasks(self) -> List[str]:
        return ["text-generation", "chat", "structured-generation"]

    @property
    def is_initialized(self) -> bool:
        return self.engine is not None

    
