from __future__ import annotations

import html
import time
from typing import Optional, Dict, Any, ClassVar, Type, TypeVar, List, Union
import uuid
from h11 import Data
import pyarrow as pa
from datetime import date, datetime
import ulid
import json
from pydantic import Field
from outlines.models import BaseModel as OutlinesBaseModel
from dataclasses import dataclass
import daft
import typing
from pydantic import HttpUrl, EmailStr
from beta.data.obj.base import DataObjectManager


T = TypeVar("T", bound="DataObject")


class DataObjectError(Exception):
    """Base exception class for CustomModel errors."""


class ULIDValidationError(DataObjectError):
    """Raised when ULID validation fails."""


class ArrowConversionError(DataObjectError):
    """Raised when conversion to/from Arrow fails."""


@dataclass
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
        type_mapping = {
            int: pa.int64(),
            float: pa.float64(),
            bool: pa.bool_(),
            str: pa.string(),
            bytes: pa.binary(),
            datetime: pa.timestamp("ms"),
            date: pa.date32(),
            time: pa.time64("us"),
            uuid.UUID: pa.string(),
            ulid.ulid: pa.string(),
            HttpUrl: pa.string(),
            EmailStr: pa.string(),
            html.Html: pa.string(),  # pa.html()
            Dict[str, str]: pa.map_(pa.string(), pa.string()),
            Dict[str, int]: pa.map_(pa.string(), pa.int64()),
            Dict[str, float]: pa.map_(pa.string(), pa.float64()),
            Dict[str, bool]: pa.map_(pa.string(), pa.bool_()),
            Dict[str, bytes]: pa.map_(pa.string(), pa.binary()),
            Dict[str, datetime]: pa.map_(pa.string(), pa.timestamp("ms")),
            Dict[str, date]: pa.map_(pa.string(), pa.date32()),
            Dict[str, time]: pa.map_(pa.string(), pa.time64("us")),
            Dict[str, uuid.UUID]: pa.map_(pa.string(), pa.string()),
            Dict[str, ulid.ulid]: pa.map_(pa.string(), pa.string()),
            Dict[str, Any]: pa.map_(pa.string(), pa.json()),
            List[str]: pa.list_(pa.string()),
            List[int]: pa.list_(pa.int64()),
            List[float]: pa.list_(pa.float64()),
            List[bool]: pa.list_(pa.bool_()),
            List[bytes]: pa.list_(pa.binary()),
            List[datetime]: pa.list_(pa.timestamp("ms")),
            List[date]: pa.list_(pa.date32()),
            List[time]: pa.list_(pa.time64("us")),
            List[uuid.UUID]: pa.list_(pa.string()),
            List[ulid.ulid]: pa.list_(pa.string()),
            List[Any]: pa.list_(pa.json()),
            List[HttpUrl]: pa.list_(pa.string()),
            List[EmailStr]: pa.list_(pa.string()),
            List[html.Html]: pa.list_(pa.string()),
            List[Dict[str, str]]: pa.list_(pa.map_(pa.string(), pa.string())),
            List[Dict[str, int]]: pa.list_(pa.map_(pa.string(), pa.int64())),
            List[Dict[str, float]]: pa.list_(pa.map_(pa.string(), pa.float64())),
            List[Dict[str, bool]]: pa.list_(pa.map_(pa.string(), pa.bool_())),
            List[Dict[str, bytes]]: pa.list_(pa.map_(pa.string(), pa.binary())),
            typing.Any: pa.string(),  # Default fallback for unspecified types # Wont this be a problem? reply: #
        }
        return type_mapping.get(pydantic_type, pa.string())

    def to_arrow(self) -> pa.RecordBatch:
        try:
            data = self.dict()
            fields = []
            arrays = []

            for field_name, field_value in data.items():
                python_type = (
                    type(field_value) if field_value is not None else type(None)
                )
                arrow_type = self._get_arrow_type(python_type)

                fields.append(pa.field(field_name, arrow_type))

                if isinstance(field_value, (dict, list)):
                    arrays.append(pa.array([json.dumps(field_value)]))
                elif isinstance(
                    field_value,
                    (
                        datetime,
                        date,
                        time,
                        uuid.UUID,
                        ulid.ulid,
                        HttpUrl,
                        EmailStr,
                    ),
                ):
                    arrays.append(pa.array([str(field_value)]))
                else:
                    arrays.append(pa.array([field_value]))

            return pa.Table.from_arrays(arrays, schema=pa.schema(fields))
        except Exception as e:
            raise ArrowConversionError(f"Error converting to Arrow: {str(e)}") from e

    @classmethod
    def from_arrow(cls, table: pa.RecordBatch) -> DataObject:
        try:
            data = table.to_pydict()
            for field_name, field_type in cls.__annotations__.items():
                if field_name in data:
                    if field_type == datetime:
                        data[field_name] = datetime.fromisoformat(data[field_name][0])
                    elif field_type == date:
                        data[field_name] = date.fromisoformat(data[field_name][0])
                    elif field_type == time:
                        data[field_name] = time.fromisoformat(data[field_name][0])
                    elif field_type == uuid.UUID:
                        data[field_name] = uuid.UUID(data[field_name][0])
                    elif field_type == ulid.ulid:
                        data[field_name] = ulid.parse(data[field_name][0])
                    elif field_type in (
                        Dict[str, Any],
                        List[Any],
                        List[Dict[str, Any]],
                    ):
                        data[field_name] = json.loads(data[field_name][0])
                    elif field_type == HttpUrl:
                        data[field_name] = HttpUrl(data[field_name][0])
                    elif field_type == EmailStr:
                        data[field_name] = EmailStr(data[field_name][0])
            return cls(**data)
        except Exception as e:
            raise ArrowConversionError(f"Error converting from Arrow: {str(e)}") from e

    def to_json_schema(self) -> Dict[str, Any]:
        return json.loads(self.schema_json())

    @classmethod
    def from_json(cls: Type[T], json_data: str) -> T:
        return cls.parse_raw(json_data)

    def to_json(self) -> str:
        return self.json()

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
    def from_daft_dataframe(cls: Type[T], df: daft.DataFrame) -> List[T]:
        arrow_table = df.to_arrow()
        return [
            cls.from_arrow(record_batch) for record_batch in arrow_table.to_batches()
        ]

    def to_daft_dataframe(self) -> daft.DataFrame:
        return daft.from_arrow(self.to_arrow())

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

    @classmethod
    def save_data_objects(cls, objects: List[T], namespace: str, table_name: str):
        """Save a list of DataObjects to a DeltaCat dataset."""
        cls.manager.save_data_objects(objects, namespace, table_name)

    @classmethod
    def get_data_objects(cls, namespace: str, table_name: str) -> List[T]:
        """Retrieve a list of DataObjects from a DeltaCat dataset."""
        return cls.manager.get_data_objects(namespace, table_name)

    @classmethod
    def delete_data_objects(cls, namespace: str, table_name: str):
        """Delete a list of DataObjects from a DeltaCat dataset."""
        cls.manager.delete_data_objects(namespace, table_name)

    @classmethod
    def update_data_object(cls, data_object: T, namespace: str, table_name: str):
        """Update a single DataObject in a DeltaCat dataset."""
        cls.manager.update_data_object(data_object, namespace, table_name)
