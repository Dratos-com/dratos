from __future__ import annotations
import asyncio
from typing import Protocol, List, Any, Optional, Union, Dict
from enum import Enum
from datetime import datetime
import logging
from daft import DataFrame, Schema, DataType, lit, col, udf
import daft
import lancedb
from lancedb.embeddings import get_registry
import numpy as np
from pydantic import BaseModel, Field
from ulid import ULID
import gzip
import uuid
import spacy


# Define the node types for content
class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    THREE_D = "3d"
    GRAPH = "graph"


class ContentNode(BaseModel):
    content_type: ContentType
    content: Union[str, bytes]  # Text or binary data depending on the content type
    metadata: Optional[Dict[str, Any]] = None


class Edge(BaseModel):
    from_node: str  # ID of the source node
    to_node: str  # ID of the destination node
    metadata: Optional[Dict[str, Any]] = None


class SchemaWrapper(BaseModel):
    schema: Any

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if not isinstance(value, Schema):
            raise ValueError("Must be a Schema instance")
