from dynaconf import Dynaconf
from pydantic import BaseSettings
from typing import Optional


class LocalStorageContext(BaseSettings):
    local_path: Optional[str] = "~/lancedb"
    anonymous: Optional[bool] = False

    class Config:
        env_prefix = "DRATOS_STORAGE_LOCAL_"
        env_file = ".env"


class S3StorageContext(BaseSettings):
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

    class Config:
        env_prefix = "DRATOS_STORAGE_AWS_"
        env_file = ".env"


class GCSStorageContext(BaseSettings):
    project_id: Optional[str] = None
    credentials: Optional[str] = None  # Path to credentials JSON or JSON string
    token: Optional[str] = None
    anonymous: Optional[bool] = False

    # Additional arguments from LanceDB
    google_service_account: Optional[str] = None
    google_service_account_key: Optional[str] = None

    class Config:
        env_prefix = "DRATOS_STORAGE_GCS_"
        env_file = ".env"


class AzureStorageContext(BaseSettings):
    storage_account: Optional[str] = None
    access_key: Optional[str] = None
    sas_token: Optional[str] = None
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    anonymous: Optional[bool] = False

    # Additional arguments from LanceDB
    storage_account_key: Optional[str] = None
    storage_sas_key: Optional[str] = None
    storage_token: Optional[str] = None
    storage_use_emulator: Optional[bool] = False
    endpoint: Optional[str] = None
    use_fabric_endpoint: Optional[bool] = False
    msi_endpoint: Optional[str] = None
    object_id: Optional[str] = None
    msi_resource_id: Optional[str] = None
    federated_token_file: Optional[str] = None
    use_azure_cli: Optional[bool] = False
    disable_tagging: Optional[bool] = False

    class Config:
        env_prefix = "DRATOS_STORAGE_AZURE_"
        env_file = ".env"


class StorageContext:
    s3: Optional[S3StorageContext] = None
    gcs: Optional[GCSStorageContext] = None
    azure: Optional[AzureStorageContext] = None
    local: Optional[LocalStorageContext] = None

    def __init__(self, settings: Dynaconf):
        # Initialize dynaconf settings
        if settings.get("storage.s3.access_key_id"):
            self.s3 = S3StorageContext(**(settings.get("storage.s3") or {}))
        elif settings.get("storage.gcs.project_id"):
            self.gcs = GCSStorageContext(**(settings.get("storage.gcs") or {}))
        elif settings.get("storage.azure.storage_account"):
            self.azure = AzureStorageContext(**(settings.get("storage.azure") or {}))
        elif settings.get("storage.local.local_path"):
            self.local = LocalStorageContext(**(settings.get("storage.local") or {}))
        else:
            raise ValueError("No valid StorageContext found.")
