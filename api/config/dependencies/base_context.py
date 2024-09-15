from abc import ABC, abstractmethod
from api.config.resources.storage_config import StorageConfig
from api.config.resources.compute_config import ComputeConfig


class BaseContext(ABC):
    """
    Base Context class for all client contexts.
    """

    name: str
    version: int

    @abstractmethod
    def configure_storage(self, storage_config: StorageConfig):
        pass

    @abstractmethod
    def configure_compute(self, compute_config: ComputeConfig):
        pass

    @abstractmethod
    def get_context(self):
        pass

    @abstractmethod
    def set_context(self, context):
        pass

    @abstractmethod
    def update_context(self, context):
        pass
