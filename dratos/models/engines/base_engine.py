"""Base classes for all engines."""

import os
import json
import logging
from abc import ABC, abstractmethod

from typing import Any, Dict, List, AsyncIterator

from pydantic import BaseModel

class BaseEngine(ABC):
    """Base class for all engines."""
    def __init__(
        self,
        engine: str,
    ):
        self._is_initialized = False

        is_test_env = os.getenv("IS_TEST_ENV")
        
        if is_test_env == 'true':
            # Write ENGINE to a shared file
            shared_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'test_server', 'shared_env.json')
            with open(shared_file, 'w') as f:
                json.dump({"ENGINE": engine}, f)

            logging.info("\033[94mTEST ENV SELECTED\033[0m")
            self.base_url = os.getenv("TEST_API_BASE_URL")
            self.api_key = "TEST_API_KEY"

    @abstractmethod
    def initialize(self, asynchronous: bool = False) -> None:
        """Initialize the model and any necessary resources.""" 

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the adapter and free resources."""

    @abstractmethod
    def sync_gen(
        self,
        model_name: str = None,
        response_model: BaseModel | None = None,
        tools: List[Dict] = None,
        messages: List[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Generate text using the adapter."""

    @abstractmethod
    async def async_gen(
        self,
        model_name: str = "gpt-4",
        response_model: BaseModel | None = None,
        tools: List[Dict] = None,
        messages: List[Dict] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate text using the engine asynchronously."""

    @abstractmethod
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for the adapter."""
   
    @abstractmethod
    def support_tools(self, model_name: str) -> bool:
        """Check if the adapter supports tools."""

    @abstractmethod
    def supported_documents(self, model_name: str, extension: str) -> bool:
        """Check if the adapter supports a specific document type."""

    @abstractmethod
    def support_structured_output(self, model_name: str) -> bool:
        """Check if the adapter supports structured output."""
