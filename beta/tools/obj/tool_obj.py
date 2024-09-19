from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
from pydantic import BaseModel
from typing import Any, Dict
from beta.data.obj import DataObject


class Tool(DataObject):
    name: str

    __schema__ = "obj"
    __tablename__ = "functions"

    def __call__(self, data: DataObject):
        return self._udf(data)

    def _udf(self, data: DataObject):
        # Define a UDF (e.g., text vectorization)
        text = data.content["text"]
        vectors = self._vectorize(text)
        data.vectors = vectors
        return vectors

    def _vectorize(self, text: str):
        # Placeholder vectorization (this should be your implementation)
        return [ord(c) for c in text]  # Example: ASCII value of each char


# Example usage
tool = Tool(name="Vectorizer")
