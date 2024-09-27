from __future__ import annotations

import hashlib
import gzip
import mimetypes
import os
from typing import Any, ClassVar, Dict, List, Optional

import daft
import pyarrow as pa
from daft import col
from daft.io import IOConfig
from ulid import ULID


artifact_schema = pa.schema(
    [
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
)


class Artifact:
    """
    Dataframe model for managing artifacts.
    Utilizes Daft's DataFrame for efficient data manipulation and scalability.
    Integrates with Ray for distributed processing.
    """

    schema: ClassVar[pa.Schema] = artifact_schema
    obj_type: ClassVar[str] = "Artifact"

    def __init__(
        self,
        files: Optional[List[str]] = None,
        df: Optional[daft.DataFrame] = None,
        io_config: Optional[IOConfig] = IOConfig(),
        uri_prefix: Optional[str] = "",
    ):
        self.io_config = io_config
        self.uri_prefix = uri_prefix
        self.df = df

        if df is None:
            self.df = daft.from_arrow(artifact_schema.empty_table())
            if files:
                self.df = self.from_files(files, uri_prefix, self.io_config)
        else:
            self.add_files(files, uri_prefix, self.io_config)

        self.df = self.validate_schema(self.df)

    def get_artifact_uri(
        self,
        custom_prefix: Optional[str] = None,
        io_config: Optional[IOConfig] = None,
    ) -> str:
        """
        Constructs the artifact URI prefix based on the storage backend configurations.

        Parameters:
            custom_prefix (Optional[str]): Optional prefix for the URI.
            io_config (Optional[daft.io.IOConfig]): Optional IOConfig for the URI.

        Returns:
            str: The constructed artifact URI prefix.
        """
        if io_config:
            if io_config.s3.access_key:
                s3 = io_config.s3
                if not hasattr(s3, "bucket") or not s3.bucket:
                    raise ValueError("S3Config must have a 'bucket' defined.")
                prefix = s3.prefix + custom_prefix if custom_prefix else s3.prefix
                return f"s3://{s3.bucket}/{prefix}/" if prefix else f"s3://{s3.bucket}/"

            elif io_config.gcs.project_id:
                gcs = io_config.gcs
                if not hasattr(gcs, "bucket") or not gcs.bucket:
                    raise ValueError("GCSConfig must have a 'bucket' defined.")
                prefix = gcs.prefix + custom_prefix if custom_prefix else gcs.prefix
                return (
                    f"gs://{gcs.bucket}/{prefix}/" if prefix else f"gs://{gcs.bucket}/"
                )

            elif io_config.azure.access_key:
                azure = io_config.azure
                if not hasattr(azure, "container") or not azure.container:
                    raise ValueError("AzureConfig must have a 'container' defined.")
                prefix = azure.prefix + custom_prefix if custom_prefix else azure.prefix
                return (
                    f"az://{azure.container}/{prefix}/"
                    if prefix
                    else f"az://{azure.container}/"
                )

        else:
            return f"{custom_prefix}/" if custom_prefix else "/"

    def from_files(
        self,
        files: List[str],
        custom_prefix: Optional[str] = "",
        io_config: Optional[IOConfig] = None,
    ) -> daft.DataFrame:
        """
        Create an Artifact instance from a list of files.
        
        Args:
            files (List[str]): List of file paths.
            custom_prefix (Optional[str]): Custom prefix for the artifact URI.
            io_config (Optional[IOConfig]): I/O configuration.
        
        Returns:
            daft.DataFrame: A DataFrame containing the artifact data.
        """
        rows = []
        for file in files:
            with open(file, "rb") as f:
                content = f.read()

            ulid_id = ULID()
            file_id = str(ulid_id)
            now = ulid_id.datetime
            file_name = os.path.basename(file)
            file_name = file_name.replace(" ", "_")
            artifact_uri = self.get_artifact_uri(custom_prefix, io_config)
            artifact_uri = f"{artifact_uri}{file_id}__{file_name}"

            new_row = {
                "id": file_id,
                "type": self.obj_type,
                "created_at": now,
                "updated_at": now,
                "inserted_at": now,
                "name": file_name,
                "artifact_uri": artifact_uri,
                "payload": content,
                "extension": os.path.splitext(file)[1][1:],
                "mime_type": mimetypes.guess_type(file)[0] or "application/x-gzip",
                "version": "1.0",
                "size_bytes": len(content),
                "checksum": hashlib.md5(content).hexdigest(),
            }
            rows.append(new_row)

        df = daft.from_arrow(self.schema.empty_table())
        df = df.concat(daft.from_pylist(rows))
        df.collect()
        return df

    def add_files(
        self,
        files: List[str],
        uri_prefix: Optional[str] = "",
        io_config: Optional[IOConfig] = None,
    ) -> None:
        """
        Adds new files to the Artifact DataFrame.

        Parameters:
            files (List[str]): List of file paths to add.
            uri_prefix (Optional[str]): Prefix for the artifact URI.
        """
        new_df = self.from_files(files, uri_prefix, io_config)
        self.df = self.df.concat(new_df).collect()

    def filter_by_type(self, obj_type: str) -> daft.DataFrame:
        """
        Filters artifacts by their type.

        Parameters:
            obj_type (str): The type to filter by.

        Returns:
            daft.DataFrame: Filtered DataFrame.
        """
        return self.df.where(col("type") == obj_type)

    def get_artifact_by_id(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves an artifact by its ID.

        Parameters:
            artifact_id (str): The ULID of the artifact.

        Returns:
            Optional[Dict[str, Any]]: Artifact data or None if not found.
        """
        filtered_df = self.df.where(col("id") == artifact_id)
        if filtered_df.count() == 0:
            return None
        return filtered_df.collect().to_pydict()[0]

    def verify_checksum(self, artifact_id: str) -> bool:
        """
        Verifies the checksum of an artifact.

        Parameters:
            artifact_id (str): The ULID of the artifact.

        Returns:
            bool: True if checksum matches, False otherwise.
        """
        artifact = self.get_artifact_by_id(artifact_id)
        if not artifact:
            return False
        actual_payload = gzip.decompress(artifact["payload"])
        actual_checksum = hashlib.md5(actual_payload).hexdigest()
        return actual_checksum == artifact["checksum"]

    def validate_schema(self, df: daft.DataFrame) -> daft.DataFrame:
        """
        Validates that the DataFrame adheres to the defined schema.

        Parameters:
            df (daft.DataFrame): The DataFrame to validate.

        Returns:
            daft.DataFrame: The validated DataFrame.

        Raises:
            ValueError: If the schema does not match.
        """
        
        if df.schema != daft.Schema.from_pyarrow_schema(self.schema):
            raise ValueError("DataFrame schema does not match the Artifact schema.")
        return df

    def repartition_by_date(self, num_partitions: int) -> None:
        """
        Repartitions the DataFrame based on the 'created_at' timestamp.

        Parameters:
            num_partitions (int): Number of partitions to create.
        """
        self.df = self.df.repartition(num_partitions, by=["created_at"])

    def export_to_parquet(self, path: str) -> None:
        """
        Exports the DataFrame to Parquet files.

        Parameters:
            path (str): Destination path for Parquet files.
        """
        self.df.write_parquet(path, io_config=self.io_config)

    def __repr__(self) -> str:
        return f"<Artifact with {self.df.count()} files>"

    def __str__(self) -> str:
        return self.__repr__()
