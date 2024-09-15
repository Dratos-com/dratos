from typing import Optional
from beta.data.obj.data_object import DataObject
import os
import yaml


class S3StorageConfig(DataObject):
    # Common arguments
    region_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    anonymous: Optional[bool] = False

    # Additional arguments from Daft
    expiry: Optional[int] = None
    buffer_time: Optional[int] = None
    max_connections: Optional[int] = None
    retry_initial_backoff_ms: Optional[int] = None
    connect_timeout_ms: Optional[int] = None
    read_timeout_ms: Optional[int] = None
    num_tries: Optional[int] = None
    retry_mode: Optional[str] = None
    use_ssl: Optional[bool] = True
    verify_ssl: Optional[bool] = True
    check_hostname_ssl: Optional[bool] = True
    requester_pays: Optional[bool] = False
    force_virtual_addressing: Optional[bool] = False
    profile_name: Optional[str] = None

    # Additional arguments from LanceDB
    virtual_hosted_style: Optional[bool] = False
    s3_express: Optional[bool] = False
    server_side_encryption: Optional[bool] = False
    sse_kms_key_id: Optional[str] = None
    sse_bucket_key_enabled: Optional[bool] = False


class GCSStorageConfig(DataObject):
    project_id: Optional[str] = None
    credentials: Optional[str] = None  # Path to credentials JSON or JSON string
    token: Optional[str] = None
    anonymous: Optional[bool] = False

    # Additional arguments from LanceDB
    google_service_account: Optional[str] = None
    google_service_account_key: Optional[str] = None


class AzureStorageConfig(DataObject):
    storage_account: Optional[str] = None
    access_key: Optional[str] = None
    sas_token: Optional[str] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    anonymous: Optional[bool] = False

    # Additional arguments from LanceDB
    storage_account_key: Optional[str] = None
    azure_storage_sas_key: Optional[str] = None
    azure_storage_token: Optional[str] = None
    azure_storage_use_emulator: Optional[bool] = False
    azure_endpoint: Optional[str] = None
    azure_use_fabric_endpoint: Optional[bool] = False
    azure_msi_endpoint: Optional[str] = None
    azure_object_id: Optional[str] = None
    azure_msi_resource_id: Optional[str] = None
    azure_federated_token_file: Optional[str] = None
    azure_use_azure_cli: Optional[bool] = False
    azure_disable_tagging: Optional[bool] = False


class LocalStorageConfig(DataObject):
    local_path: Optional[str] = "~/lancedb"
    anonymous: Optional[bool] = False
    # Additional local storage config options can be added here


class StorageConfig(DataObject):
    def __init__(self, config_path: str = None):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        self.config_dict = config_dict["storage"]

        self.s3: Optional[S3StorageConfig] = None
        self.gcs: Optional[GCSStorageConfig] = None
        self.azure: Optional[AzureStorageConfig] = None
        self.local: Optional[LocalStorageConfig] = None

        self._configure_environment_variables()

    def _configure_environment_variables(self):
        """Set environment variables based on storage configurations."""

        if os.getenv("AWS_ACCESS_KEY_ID"):
            self.s3 = S3StorageConfig(**self.config_dict["s3"])
            self.s3.access_key_id = os.getenv(
                "AWS_ACCESS_KEY_ID", self.s3.access_key_id
            )
            self.s3.secret_access_key = os.getenv(
                "AWS_SECRET_ACCESS_KEY", self.s3.secret_access_key
            )
            self.s3.session_token = os.getenv(
                "AWS_SESSION_TOKEN", self.s3.session_token
            )
            self.s3.region_name = os.getenv("AWS_REGION", self.s3.region_name)
            self.s3.endpoint_url = os.getenv("AWS_ENDPOINT_URL", self.s3.endpoint_url)
            self.s3.anonymous = os.getenv("AWS_ANONYMOUS", self.s3.anonymous)
            self.s3.expiry = os.getenv("AWS_EXPIRY", self.s3.expiry)
            self.s3.buffer_time = os.getenv("AWS_BUFFER_TIME", self.s3.buffer_time)
            self.s3.max_connections = os.getenv(
                "AWS_MAX_CONNECTIONS", self.s3.max_connections
            )
            self.s3.retry_initial_backoff_ms = os.getenv(
                "AWS_RETRY_INITIAL_BACKOFF_MS", self.s3.retry_initial_backoff_ms
            )
            self.s3.connect_timeout_ms = os.getenv(
                "AWS_CONNECT_TIMEOUT_MS", self.s3.connect_timeout_ms
            )
            self.s3.read_timeout_ms = os.getenv(
                "AWS_READ_TIMEOUT_MS", self.s3.read_timeout_ms
            )
            self.s3.num_tries = os.getenv("AWS_NUM_TRIES", self.s3.num_tries)
            self.s3.retry_mode = os.getenv("AWS_RETRY_MODE", self.s3.retry_mode)
            self.s3.use_ssl = os.getenv("AWS_USE_SSL", self.s3.use_ssl)
            self.s3.verify_ssl = os.getenv("AWS_VERIFY_SSL", self.s3.verify_ssl)

        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            self.gcs = GCSStorageConfig(**self.config_dict["gcs"])
            self.gcs.project_id = os.getenv("GOOGLE_PROJECT_ID", self.gcs.project_id)
            self.gcs.credentials = os.getenv(
                "GOOGLE_APPLICATION_CREDENTIALS", self.gcs.credentials
            )
            self.gcs.anonymous = os.getenv("GOOGLE_ANONYMOUS", self.gcs.anonymous)
            self.gcs.google_service_account = os.getenv(
                "GOOGLE_SERVICE_ACCOUNT", self.gcs.google_service_account
            )
            self.gcs.google_service_account_key = os.getenv(
                "GOOGLE_SERVICE_ACCOUNT_KEY", self.gcs.google_service_account_key
            )

        elif os.getenv("AZURE_STORAGE_ACCESS_KEY"):
            self.azure = AzureStorageConfig(**self.config_dict["azure"])
            self.azure.access_key = os.getenv(
                "AZURE_STORAGE_ACCESS_KEY", self.azure.access_key
            )
            self.azure.sas_token = os.getenv(
                "AZURE_STORAGE_SAS_TOKEN", self.azure.sas_token
            )
            self.azure.tenant_id = os.getenv("AZURE_TENANT_ID", self.azure.tenant_id)
            self.azure.client_id = os.getenv("AZURE_CLIENT_ID", self.azure.client_id)
            self.azure.client_secret = os.getenv(
                "AZURE_CLIENT_SECRET", self.azure.client_secret
            )
            self.azure.anonymous = os.getenv("AZURE_ANONYMOUS", self.azure.anonymous)
            self.azure.storage_account_key = os.getenv(
                "AZURE_STORAGE_ACCOUNT_KEY", self.azure.storage_account_key
            )
            self.azure.azure_storage_token = os.getenv(
                "AZURE_STORAGE_TOKEN", self.azure.azure_storage_token
            )
            self.azure.azure_storage_use_emulator = os.getenv(
                "AZURE_STORAGE_USE_EMULATOR", self.azure.azure_storage_use_emulator
            )
            self.azure.azure_endpoint = os.getenv(
                "AZURE_ENDPOINT", self.azure.azure_endpoint
            )
            self.azure.azure_use_fabric_endpoint = os.getenv(
                "AZURE_USE_FABRIC_ENDPOINT", self.azure.azure_use_fabric_endpoint
            )
            self.azure.azure_msi_endpoint = os.getenv(
                "AZURE_MSI_ENDPOINT", self.azure.azure_msi_endpoint
            )
            self.azure.azure_object_id = os.getenv(
                "AZURE_OBJECT_ID", self.azure.azure_object_id
            )
            self.azure.azure_msi_resource_id = os.getenv(
                "AZURE_MSI_RESOURCE_ID", self.azure.azure_msi_resource_id
            )
            self.azure.azure_federated_token_file = os.getenv(
                "AZURE_FEDERATED_TOKEN_FILE", self.azure.azure_federated_token_file
            )
            self.azure.azure_use_azure_cli = os.getenv(
                "AZURE_USE_AZURE_CLI", self.azure.azure_use_azure_cli
            )
            self.azure.azure_disable_tagging = os.getenv(
                "AZURE_DISABLE_TAGGING", self.azure.azure_disable_tagging
            )

        else:
            self.local = LocalStorageConfig(**self.config_dict["local"])
