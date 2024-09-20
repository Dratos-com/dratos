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
from ulid import ULID
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import os
import json
import daft
from daft import lit, col

from beta.data.obj.base.data_object import DataObject, data_object_schema

artifact_schema = [
    pa.field(
        "id",
        pa.string(),
        nullable=False,
        metadata={"description": "ULID Unique identifier for the record"},
    ),
    pa.field(
        "type",
        pa.string(),
        nullable=False,
        metadata={"description": "Type of the data object"},
    ),
    pa.field(
        "created_at",
        pa.timestamp("ns", tz="UTC"),
        nullable=False,
        metadata={"description": "Timestamp when the record was created"},
    ),
    pa.field(
        "updated_at",
        pa.timestamp("ns", tz="UTC"),
        nullable=False,
        metadata={"description": "Timestamp when the record was last updated"},
    ),
    pa.field(
        "inserted_at",
        pa.timestamp("ns", tz="UTC"),
        nullable=False,
        metadata={
            "description": "Timestamp when the data object was inserted into the database"
        },
    ),
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
]


class Artifact:
    """
    Versatile model for managing artifacts in a data system.
    """

    schema: ClassVar[pa.Schema] = artifact_schema
    obj_type: ClassVar[str] = "Artifact"

    def __init__(self, files: List[str] = None, uri_prefix: Optional[str] = None):
        # Create an empty DataFrame with the schema from ArtifactObject
        empty_data = {field.name: [] for field in self.schema}
        self.df = daft.from_pydict(empty_data)

        if files:
            self._populate_from_file(files, uri_prefix)

    def _populate_from_file(self, files: List[str], uri_prefix: Optional[str] = None):
        rows = []
        for file in files:
            with open(file, "rb") as f:
                content = f.read()

            file_id = str(ULID())
            now = datetime.now().isoformat()
            file_name = os.path.basename(file)

            new_row = {
                "id": file_id,
                "type": "file",
                "created_at": now,
                "updated_at": now,
                "inserted_at": now,
                "name": file_name,
                "artifact_uri": (
                    f"{uri_prefix}/{file_id}__{file_name}.gzip" if uri_prefix else ""
                ),
                "payload": gzip.compress(content),
                "extension": os.path.splitext(file)[1][1:],
                "mime_type": mimetypes.guess_type(file)[0]
                or "application/octet-stream",
                "version": "1.0",
                "size_bytes": len(content),
                "checksum": hashlib.md5(content).hexdigest(),
            }
            rows.append(new_row)

        self.df = daft.from_pylist(rows)

    def __repr__(self):
        return f"<Artifact with {len(self.df)} files>"

    def __str__(self):
        return self.__repr__()
