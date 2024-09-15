from __future__ import annotations


import time
from typing import Optional, Dict, Any, ClassVar, Type, TypeVar, List, Union
import uuid
import pyarrow as pa
from datetime import timezone, datetime
from ulid import ULID
import json
import daft
import typing
from pydantic import BaseModel, Field
from pydantic_to_pyarrow import get_pyarrow_schema
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from api.config.dependencies.lance import LanceContext


class DataObjectError(Exception):
    """Base exception class for CustomModel errors."""


class ArrowConversionError(DataObjectError):
    """Raised when conversion to/from Arrow fails."""


class DataObject(LanceModel):
    """
    Base class for all data objects in the system.
    Provides common fields and methods for Arrow conversion and data manipulation.
    """

    __tablename__: ClassVar[str] = "objects"
    __namespace__: ClassVar[str] = "data"

    id: str = Field(
        default_factory=lambda: str(ULID()), description="ULID Unique identifier"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="UTC timestamp of last update"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )
    embedding: Optional[List[Vector]] = Field(
        default=None, description="Embedding vectors"
    )

    # Get Timestamp from string ULID
    def get_timestamp_from_ulid(self) -> datetime:
        return datetime.fromtimestamp(
            ULID.from_str(self.id).timestamp(), tz=timezone.utc
        )

    @property
    def created_at(self) -> datetime:
        """Return the creation time based on the ULID."""
        # convert ULID to Timestamp in UTC
        return self.get_timestamp_from_ulid()

    def update_timestamp(self):
        """Update the 'updated_at' timestamp to the current UTC time."""
        self.updated_at = datetime.now(timezone.utc)

    @classmethod
    def get_schema(self) -> Dict[str, Any]:
        return self.model_json_schema()

    @classmethod
    def get_arrow_schema(cls) -> pa.Schema:
        """Generate an Arrow schema based on the class fields."""
        try:
            return get_pyarrow_schema(cls)
        except Exception as e:
            raise ArrowConversionError(f"Failed to convert to Arrow schema: {e}")

    @classmethod
    def get_daft_schema(cls) -> daft.Schema:
        """Generate a Daft schema based on the class fields."""
        try:
            return daft.Schema.from_arrow_schema(cls.get_arrow_schema())
        except Exception as e:
            raise ArrowConversionError(f"Failed to convert to Daft schema: {e}")
        
    @classmethod
    def get_lancedb_schema(cls) -> Dict[str, Any]:
        """Generate a LanceDB schema based on the class fields."""
        return cls.get_schema()
    

class DataObjectTable:
    """
    Table for DataObject subclasses who inherit the __tablename__ and __namespace__ class variables.
    """
    
    def __init__(self, storage_context: StorageContext):
        """
        Initialize the DataObjectTable.
        """
        self.dob = dob
        self.db = db
        self.location = location
        self.schema = None
        self.table = None

        if self.location is None:
            self.location = f"{dob.__namespace__}/{dob.__tablename__}"

        data_object_schema = pa.schema([
            pa.field("id", pa.string(), nullable=False, primary_key=True),
            pa.field("metadata", pa.map(pa.string(), pa.string()), nullable=True),
            pa.field("object_dump", pa.struct(), nullable=False, primary_key=False),
            pa.field("object_type", pa.string(), nullable=False, primary_key=False),
            pa.field("object_schema", pa.schema(), nullable=False, primary_key=False),
        ])

        timestamp_schema = pa.schema([
            pa.field("updated_at", pa.timestamp(timezone=True), nullable=True),
            pa.field("created_at", pa.timestamp(timezone=True), nullable=False),
            pa.field("inserted_at", pa.timestamp(timezone=True), nullable=False)
        ])

        session_schema = pa.schema([
            pa.field("user_id", pa.string(), nullable=False, primary_key=False),
            pa.field("account_id", pa.string(), nullable=False, primary_key=False),
            pa.field("session_id", pa.string(), nullable=False, primary_key=False),
            pa.field("region_id", pa.string(), nullable=False, primary_key=False),
        ])

        governance_schema = pa.schema([
            pa.field("is_ai_generated", pa.bool(), nullable=False, primary_key=False),
            pa.field("sensitivity", pa.string(), nullable=False, primary_key=False),
        ])

        multi_index_schema = pa.schema([
            pa.field("indices", pa.list(pa.struct()), nullable=False, primary_key=False),
        ])

         vector_schema = pa.schema([
            pa.field("vector", pa.list(pa.float32()), nullable=False, primary_key=False),
        ])


        self.schema = data_object_schema \
            .append(timestamp_schema) \
            .append(session_schema) \
            .append(multi_index_schema) \
            .append(vector_schema)
            
        if self.table is None:
            self.table = await self.db.create_table(self.location, schema=self.schema)
        else:
            self.table = self.db.get_table(self.location)

        

        self.table = pa.Table.from_arrays([], schema=index_schema)

