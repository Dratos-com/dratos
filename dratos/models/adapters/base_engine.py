"""Base classes for all engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, TYPE_CHECKING
import pyarrow as pa
from outlines.processors import BaseLogitsProcessor

if TYPE_CHECKING:
    from .openai_engine import OpenAIEngineConfig


class BaseEngineConfig(ABC):
    """Base class for all engine configs."""
    api_key: str
    base_url: str


class BaseEngine(ABC):
    """Base class for all engines."""
    def __init__(
        self,
        config: BaseEngineConfig,
    ):
        self.config = config
        self.logits_processor: Optional[BaseLogitsProcessor] = None
        self._is_initialized = False

    @abstractmethod
    async def initialize(self, config: BaseEngineConfig) -> None:
        """Initialize the model and any necessary resources."""
        

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the engine and free resources."""
        

    @abstractmethod
    async def generate(self, prompt: Union[str, List[str]], messages: List[Dict[str, str]] = None, **kwargs) -> Any:
        """Generate text based on the input prompt."""
        

    @abstractmethod
    async def generate_structured(
        self,
        prompt: Union[str, List[str]],
        structure: Union[str, Dict, pa.Schema],
        grammar: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Any:
        """Generate structured output based on the given prompt and structure."""
        
    
    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """Return a list of tasks supported by this engine."""
    
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """Return a list of tasks supported by this engine."""
        

    def set_logits_processor(self, processor: BaseLogitsProcessor) -> None:
        """Set the logits processor for structured generation."""
        self.logits_processor = processor

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a single text."""
        pass


__all__ = [
    "BaseEngine",
    "BaseEngineConfig",
]
