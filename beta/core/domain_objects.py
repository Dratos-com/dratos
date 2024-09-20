from abc import ABC, abstractmethod

class Validatable(ABC):
    @abstractmethod
    def validate(self, validator):
        pass

class Serializable(ABC):
    @abstractmethod
    def serialize(self, serializer):
        pass

class Metadata(Validatable, Serializable):
    # Existing implementation
    ...