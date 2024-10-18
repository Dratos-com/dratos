"""Base classes for all engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseEngine(ABC):
    """Base class for all engines."""
    def __init__(
        self,
    ):
        self._is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model and any necessary resources.""" 

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the engine and free resources."""
        
    @abstractmethod
    async def generate(self, prompt: str, messages: List[Dict[str, str]] = None, **kwargs) -> Any:
        """Generate text based on the input prompt."""
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Return a list of tasks supported by this engine."""
