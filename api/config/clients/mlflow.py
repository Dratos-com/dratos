from pydantic import BaseSettings, HttpUrl
from typing import Optional, Dict, Any
from dynaconf import Dynaconf
from urllib.parse import urlparse

from api.config.context.storage_context import StorageContext


class MlflowS3Config:
    s3_endpoint


class MlflowConfig(BaseSettings):
    tracking_uri: HttpUrl
    registry_uri: Optional[HttpUrl] = None

    # Authentication settings
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None

    # S3 artifact store settings
    s3_endpoint_url: Optional[HttpUrl] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None

    # Azure Blob Storage settings
    azure_storage_connection_string: Optional[str] = None
    azure_storage_access_key: Optional[str] = None

    # GCS artifact store settings
    gcs_project_id: Optional[str] = None
    gcs_bucket_name: Optional[str] = None

    # Database settings
    db_username: Optional[str] = None
    db_password: Optional[str] = None
    db_host: Optional[str] = None
    db_port: Optional[int] = None
    db_name: Optional[str] = None

    # Additional settings
    artifact_max_size: int = 1024 * 1024 * 1024 * 1024  # 10GB default
    experiment_name_prefix: str = "dratos_"
    default_artifact_root: Optional[str] = None

    def __init__(self, sc: StorageContext, settings: Dynaconf):
        super().__init__()

        # Use Dynaconf to get settings, allowing for environment variable overrides
        self.tracking_uri = settings.
        self.registry_uri = settings.get("MLFLOW_REGISTRY_URI")
        self.artifact_location = settings.get("MLFLOW_ARTIFACT_LOCATION")

        # Authentication settings
        self.username = settings.get("MLFLOW_USERNAME")
        self.password = settings.get("MLFLOW_PASSWORD")
        self.token = settings.get("MLFLOW_TOKEN")

        # S3 settings
        self.s3_endpoint_url = settings.get("MLFLOW_S3_ENDPOINT_URL")
        self.aws_access_key_id = settings.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = settings.get("AWS_SECRET_ACCESS_KEY")

        # Azure settings
        self.azure_storage_connection_string = settings.get(
            "AZURE_STORAGE_CONNECTION_STRING"
        )
        self.azure_storage_access_key = settings.get("AZURE_STORAGE_ACCESS_KEY")

        # GCS settings
        self.gcs_project_id = settings.get("GCS_PROJECT_ID")
        self.gcs_bucket_name = settings.get("GCS_BUCKET_NAME")

        # Database settings
        self.db_username = settings.get("MLFLOW_DB_USERNAME")
        self.db_password = settings.get("MLFLOW_DB_PASSWORD")
        self.db_host = settings.get("MLFLOW_DB_HOST")
        self.db_port = settings.get("MLFLOW_DB_PORT")
        self.db_name = settings.get("MLFLOW_DB_NAME")

        # Additional settings
        self.artifact_max_size = settings.get(
            "MLFLOW_ARTIFACT_MAX_SIZE", self.artifact_max_size
        )
        self.experiment_name_prefix = settings.get(
            "MLFLOW_EXPERIMENT_NAME_PREFIX", self.experiment_name_prefix
        )
        self.default_artifact_root = settings.get(
            "MLFLOW_DEFAULT_ARTIFACT_ROOT", self.artifact_location
        )

        # Set artifact location based on storage context if not provided
        if not self.artifact_location:
            if sc.local:
                self.artifact_location = (
                    f"file://{sc.local.local_path}/mlflow-artifacts"
                )
            elif sc.s3:
                self.artifact_location = f"s3://{sc.s3.bucket_name}/mlflow-artifacts"
            elif sc.azure:
                self.artifact_location = f"wasbs://{sc.azure.container}@{sc.azure.account_name}.blob.core.windows.net/mlflow-artifacts"
            elif sc.gcs:
                self.artifact_location = f"gs://{sc.gcs.bucket_name}/mlflow-artifacts"

    def get_artifact_uri(self, run_id: str, artifact_path: str) -> str:
        """
        Generate the artifact URI for a given run and artifact path.
        """
        if self.artifact_location.startswith("file://"):
            return f"{self.artifact_location}/{run_id}/artifacts/{artifact_path}"
        elif self.artifact_location.startswith("s3://"):
            return f"{self.artifact_location}/{run_id}/artifacts/{artifact_path}"
        elif self.artifact_location.startswith("wasbs://"):
            return f"{self.artifact_location}/{run_id}/artifacts/{artifact_path}"
        elif self.artifact_location.startswith("gs://"):
            return f"{self.artifact_location}/{run_id}/artifacts/{artifact_path}"
        else:
            return f"{self.tracking_uri}/artifacts/{run_id}/{artifact_path}"

    def get_db_uri(self) -> Optional[str]:
        """
        Generate the database URI if database settings are provided.
        """
        if all(
            [
                self.db_username,
                self.db_password,
                self.db_host,
                self.db_port,
                self.db_name,
            ]
        ):
            return f"postgresql://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        return None

    def get_artifact_store_config(self) -> Dict[str, Any]:
        """
        Generate the artifact store configuration based on the storage type.
        """
        parsed_uri = urlparse(self.artifact_location)
        scheme = parsed_uri.scheme

        if scheme == "file":
            return {}
        elif scheme == "s3":
            return {
                "aws_access_key_id": self.aws_access_key_id,
                "aws_secret_access_key": self.aws_secret_access_key,
                "endpoint_url": self.s3_endpoint_url,
            }
        elif scheme == "wasbs":
            return {
                "azure_storage_connection_string": self.azure_storage_connection_string,
                "azure_storage_access_key": self.azure_storage_access_key,
            }
        elif scheme == "gs":
            return {
                "gcs_project_id": self.gcs_project_id,
                "gcs_bucket_name": self.gcs_bucket_name,
            }
        else:
            return {}

    def get_mlflow_client_kwargs(self) -> Dict[str, Any]:
        """
        Generate kwargs for MLflow client initialization.
        """
        kwargs = {"tracking_uri": self.tracking_uri}
        if self.registry_uri:
            kwargs["registry_uri"] = self.registry_uri
        if self.username and self.password:
            kwargs["username"] = self.username
            kwargs["password"] = self.password
        elif self.token:
            kwargs["token"] = self.token
        return kwargs
