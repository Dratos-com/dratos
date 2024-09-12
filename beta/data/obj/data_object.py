from __future__ import annotations


import time
from typing import Optional, Dict, Any, ClassVar, Type, TypeVar, List, Union
import uuid
import pyarrow as pa
from datetime import timezone, datetime
import ulid
import json
import daft
import typing
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from pydantic_to_pyarrow import get_pyarrow_schema


class DataObjectError(Exception):
    """Base exception class for CustomModel errors."""


class ULIDValidationError(DataObjectError):
    """Raised when ULID validation fails."""


class ArrowConversionError(DataObjectError):
    """Raised when conversion to/from Arrow fails."""


class DataObject(BaseModel):
    """
    Base class for all data objects in the system.
    Provides common fields and methods for Arrow conversion and data manipulation.
    """

    __tablename__: ClassVar[str] = "objects"
    __namespace__: ClassVar[str] = "data"

    id: str = Field(
        default_factory=lambda: ulid.ulid(), description="ULID Unique identifier"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="UTC timestamp of last update"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )

    @property
    def created_at(self) -> datetime:
        """Return the creation time based on the ULID."""
        return ulid.from_str(self.id).timestamp().datetime

    def update_timestamp(self):
        """Update the 'updated_at' timestamp to the current UTC time."""
        self.updated_at = datetime.now(timezone.utc)  # Use timezone.utc directly

    @classmethod
    def get_schema(self) -> Dict[str, Any]:
        return self.model_json_schema()

    @classmethod
    def get_arrow_schema(cls) -> pa.Schema:
        """Generate an Arrow schema based on the class fields."""
        return get_pyarrow_schema(cls)

    @classmethod
    def get_daft_schema(cls) -> daft.Schema:
        """Generate a Daft schema based on the class fields."""
        return daft.Schema.from_arrow_schema(cls.get_arrow_schema())
