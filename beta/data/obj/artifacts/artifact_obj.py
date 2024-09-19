from __future__ import annotations

import typing
from typing import ClassVar
if typing.TYPE_CHECKING:
    pass

import pyarrow as pa
import gzip
import hashlib
import mimetypes
import uuid
import ulid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import os
import json

from beta.data.obj.base.data_object import DataObject

artifact_schema = pa.schema(
    [
        pa.field(
            "name",
            pa.string(),
            nullable=False,
            metadata={"description": "Name of the artifact"},
        ),
        pa.field(
            "artifact_uri",
            pa.string(),
            nullable=False,
            metadata={"description": "URI where the artifact is stored"},
        ),
        pa.field(
            "payload",
            pa.binary(),
            nullable=False,
            metadata={"description": "Gzipped content of the artifact"},
        ),
        pa.field(
            "extension",
            pa.string(),
            nullable=False,
            metadata={"description": "File extension of the artifact"},
        ),
        pa.field(
            "mime_type",
            pa.string(),
            nullable=False,
            metadata={"description": "MIME type of the artifact"},
        ),
        pa.field(
            "version",
            pa.string(),
            nullable=False,
            metadata={"description": "Version of the artifact"},
        ),
        pa.field(
            "size_bytes",
            pa.int64(),
            nullable=False,
            metadata={"description": "Size of the artifact in bytes"},
        ),
        pa.field(
            "checksum",
            pa.string(),
            nullable=False,
            metadata={"description": "MD5 checksum of the artifact"},
        ),
        pa.field(
            "metadata",
            pa.string(),
            nullable=True,
            metadata={"description": "Additional metadata associated with the artifact"},
        ),
    ]
)




class Artifact(DataObject):
    """
    Versatile model for managing artifacts in a data system.
    """

    schema: ClassVar[pa.Schema] = artifact_schema
    obj_type: ClassVar[str] = "Artifact"

    def __init__(self, file_path: str = None):
        super().__init__()
        self.file_path = file_path
        self._populate_from_file()

    def _populate_from_file(self):
        # Read file and populate attributes
        with open(self.file_path, 'rb') as file:
            content = file.read()

        self.id = str(ulid.new())
        self.name = os.path.basename(self.file_path)
        self.artifact_uri = f"gs://your-bucket/artifacts/{self.id}/{self.name}"
        self.payload = gzip.compress(content)
        self.extension = os.path.splitext(self.file_path)[1][1:]
        self.mime_type = mimetypes.guess_type(self.file_path)[0] or 'application/octet-stream'
        self.version = "1.0"  # You might want to implement versioning logic
        self.size_bytes = len(content)
        self.checksum = hashlib.md5(content).hexdigest()
        self.is_ai_generated = self.custom_metadata.get('is_ai_generated', False)
        self.sensitivity_level = self.custom_metadata.get('sensitivity_level', "CONFIDENTIAL")
        self.context_event_id = self.custom_metadata.get('context_event_id', str(ulid.new()))
        self.updated_at = datetime.now()
        self.metadata = json.dumps(self.custom_metadata) if self.custom_metadata else None

    @classmethod
    def get_arrow_schema(cls) -> pa.Schema:
        return artifact_schema

    def to_arrow(self) -> pa.RecordBatch:
        return pa.RecordBatch.from_pydict({
            "id": [self.id],
            "name": [self.name],
            "artifact_uri": [self.artifact_uri],
            "payload": [self.payload],
            "extension": [self.extension],
            "mime_type": [self.mime_type],
            "version": [self.version],
            "size_bytes": [self.size_bytes],
            "checksum": [self.checksum],
            "is_ai_generated": [self.is_ai_generated],
            "sensitivity_level": [self.sensitivity_level],
            "context_event_id": [self.context_event_id],
            "updated_at": [self.updated_at],
            "metadata": [self.metadata]
        }, schema=artifact_schema)

    def __repr__(self):
        return f"<Artifact id={self.id} {self.name} ({self.mime_type})>"

    def __str__(self):
        return self.__repr__()


