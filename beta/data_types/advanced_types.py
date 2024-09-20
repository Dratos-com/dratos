from abc import ABC, abstractmethod

class AdvancedDataType(ABC):
    @abstractmethod
    def process(self):
        pass

class EmbeddingType(AdvancedDataType):
    def __init__(self, embedding_vector: List[float]):
        self.embedding_vector = embedding_vector

    def process(self):
        # Processing logic for embeddings
        ...

class ImageType(AdvancedDataType):
    def __init__(self, image_data: bytes):
        self.image_data = image_data

    def process(self):
        # Processing logic for images
        ...

class URLType(AdvancedDataType):
    def __init__(self, url: str):
        self.url = url

    def process(self):
        # Processing logic for URLs
        ...