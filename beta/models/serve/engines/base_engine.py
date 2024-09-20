from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, TYPE_CHECKING
import pyarrow as pa
from outlines.processors import OutlinesLogitsProcessor
from ....data import DataObject

if TYPE_CHECKING:
    from .openai_engine import OpenAIEngineConfig

class BaseEngineConfig(DataObject):
    @abstractmethod
    def update(self, new_config: Dict[str, Any]) -> None:
        """Update the engine configuration."""
        pass


class BaseEngine(ABC):
    def __init__(
        self,
        model_name: str,
        config: BaseEngineConfig = None,
    ):
        self.model_name = model_name
        self.config = config or "OpenAIEngineConfig()"  # Use string as default
        self.logits_processor: Optional[OutlinesLogitsProcessor] = None
        self._is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model and any necessary resources."""
        pass

    @abstractmethod
    async def generate(self, prompt: Union[str, List[str]], **kwargs) -> Any:
        """Generate text based on the input prompt."""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: Union[str, List[str]],
        structure: Union[str, Dict, pa.Schema],
        **kwargs,
    ) -> Any:
        """Generate structured output based on the given prompt and structure."""
        pass

    @property
    @abstractmethod
    def supported_tasks(self) -> List[str]:
        """Return a list of tasks supported by this engine."""
        pass

    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """Return the configuration of the loaded model."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the engine and free resources."""
        pass

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update the engine configuration."""
        self.config.update(new_config)

    def set_logits_processor(self, processor: OutlinesLogitsProcessor) -> None:
        """Set the logits processor for structured generation."""
        self.logits_processor = processor

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
