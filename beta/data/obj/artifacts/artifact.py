from __future__ import annotations
import pyarrow as pa
from typing import Optional, Dict, Any, ClassVar, Type, TypeVar, List
import uuid
from datetime import datetime
import ulid
import json
from pydantic import Field, HttpUrl, EmailStr
from outlines.models import BaseModel as OutlinesBaseModel

T = TypeVar("T", bound="DataObject")


class DataObjectError(Exception):
    """Base exception class for DataObject errors."""


class ArrowConversionError(DataObjectError):
    """Raised when conversion to/from Arrow fails."""


class DataObject(OutlinesBaseModel):
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

    class Config:
        arbitrary_types_allowed = True

    @property
    def created_at(self) -> datetime:
        """Return the creation time based on the ULID."""
        return ulid.parse(self.id).timestamp().datetime

    def update_timestamp(self):
        """Update the 'updated_at' timestamp to the current UTC time."""
        self.updated_at = datetime.now(datetime.timezone.utc)

    @staticmethod
    def _get_arrow_type(pydantic_type):
        """Convert Pydantic types to Arrow types."""
        # ... (keep the existing type mapping)

    @classmethod
    def get_arrow_schema(cls) -> pa.Schema:
        """Generate an Arrow schema based on the class fields."""
        return pa.schema(
            [
                (field_name, cls._get_arrow_type(field.type_))
                for field_name, field in cls.__fields__.items()
            ]
        )

    @classmethod
    def to_arrow_batch(cls: Type[T], objects: List[T]) -> pa.RecordBatch:
        """Convert a list of DataObjects to an Arrow RecordBatch."""
        try:
            data = [obj.dict() for obj in objects]
            schema = cls.get_arrow_schema()
            return pa.RecordBatch.from_pylist(data, schema=schema)
        except Exception as e:
            raise ArrowConversionError(
                f"Error converting to Arrow batch: {str(e)}"
            ) from e

    @classmethod
    def from_arrow_batch(cls: Type[T], batch: pa.RecordBatch) -> List[T]:
        """Convert an Arrow RecordBatch to a list of DataObjects."""
        try:
            data = batch.to_pylist()
            return [cls(**row) for row in data]
        except Exception as e:
            raise ArrowConversionError(
                f"Error converting from Arrow batch: {str(e)}"
            ) from e

    @classmethod
    def to_arrow_table(cls: Type[T], objects: List[T]) -> pa.Table:
        """Convert a list of DataObjects to an Arrow Table."""
        batch = cls.to_arrow_batch(objects)
        return pa.Table.from_batches([batch])

    @classmethod
    def from_arrow_table(cls: Type[T], table: pa.Table) -> List[T]:
        """Convert an Arrow Table to a list of DataObjects."""
        batches = table.to_batches()
        objects = []
        for batch in batches:
            objects.extend(cls.from_arrow_batch(batch))
        return objects

    # ... (keep other utility methods like to_json_schema, from_json, to_json, etc.)
