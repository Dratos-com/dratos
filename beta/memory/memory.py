"""
This file contains the Memory class and the MemoryStore interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class Memory:
    def __init__(self, id: str, content: str, vector: List[float]):
        self.id = id
        self.content = content
        self.vector = vector


class MemoryStore(ABC):
    @abstractmethod
    def add_memory(self, memory: Memory):
        pass

    @abstractmethod
    def search_memories(self, query_vector: List[float], limit: int = 5) -> List[Memory]:
        pass

