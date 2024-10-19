"""Base classes for all engines."""

import os
import json
import logging
from abc import ABC, abstractmethod


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
    async def initialize(self) -> None:
        """Initialize the model and any necessary resources.""" 

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the engine and free resources."""