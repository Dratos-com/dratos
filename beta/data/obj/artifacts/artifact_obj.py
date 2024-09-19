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

from beta.data.obj.base.data_object import DataObject, data_object_schema

artifact_schema = [
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

    schema: ClassVar[pa.Schema] = pa.schema(data_object_schema+artifact_schema)
    obj_type: ClassVar[str] = "Artifact"

    def __init__(self, files: List[str] = None, bucket_uri: str = None):
        self.schema = {
            "id": daft.DataType.string(),
            "type": daft.DataType.string(),
            "created_at": daft.DataType.string(),
            "updated_at": daft.DataType.string(),
            "inserted_at": daft.DataType.string(),
            "name": daft.DataType.string(),
            "artifact_uri": daft.DataType.string(),
            "payload": daft.DataType.binary(),
            "extension": daft.DataType.string(),
            "mime_type": daft.DataType.string(),
            "version": daft.DataType.string(),
            "size_bytes": daft.DataType.int64(),
            "checksum": daft.DataType.string(),
        }
        
        # Create an empty DataFrame with the schema from ArtifactObject
        empty_data = {field.name: [] for field in artifact_schema}
        self.df = daft.from_pydict(empty_data)
        
        if files:
            self._populate_from_file(files, bucket_uri)

    def _populate_from_file(self, files: List[str], bucket_uri: str = None):
        for file in files:
            with open(file, 'rb') as f:
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
                "artifact_uri": f"{bucket_uri}/{file_id}/{file_name}.gzip" if bucket_uri else "",
                "payload": gzip.compress(content),
                "extension": os.path.splitext(file)[1][1:],
                "mime_type": mimetypes.guess_type(file)[0] or 'application/octet-stream',
                "version": "1.0",
                "size_bytes": len(content),
                "checksum": hashlib.md5(content).hexdigest()
            }

            new_df = daft.from_pydict({k: [v] for k, v in new_row.items()})
            self.df = self.df.concat(new_df)

    def __repr__(self):
        return f"<Artifact with {len(self.df)} files>"

    def __str__(self):
        return self.__repr__()


