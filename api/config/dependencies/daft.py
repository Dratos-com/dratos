from typing import Optional
from datetime import datetime, timezone
from daft.io import (
    S3Config as DaftS3Config,
    S3Credentials as DaftS3Credentials,
    GCSConfig as DaftGCSConfig,
    AzureConfig as DaftAzureConfig,
)

from api.config.resources.storage_config import StorageConfig, LocalStorageConfig
from api.config.dependencies.base_context import BaseContext


# Daft IO Config
# https://www.getdaft.io/projects/docs/en/latest/api_docs/doc_gen/io_configs/daft.io.IOConfig.html


class DaftLocalConfig(LocalStorageConfig):
    pass


class DaftContext(BaseContext):
    def __init__(
        self, name: str = None, version: int = 1, storage_config: StorageConfig = None
    ):
        self.name = name
        self.version = version
        self.storage = self.configure_storage(storage_config)

    def update_context(self, storage_config: Optional[StorageConfig] = None):
        self.storage = self.configure_storage(storage_config)
        self.updated_at = datetime.now(timezone.utc)
        self.version += 1
        return self

    def configure_storage(self, sc: StorageConfig):
        """
        Configure Daft Context

        """

        if sc.s3:
            daft_s3_credentials = DaftS3Credentials(
                key_id=sc.s3.access_key_id,
                access_key=sc.s3.secret_access_key,
                session_token=sc.s3.session_token,
                expiry=sc.s3.expiry,
            )
            daft_s3_config = DaftS3Config(
                region_name=sc.s3.region_name,
                endpoint_url=sc.s3.endpoint_url,
                key_id=sc.s3.access_key_id,
                session_token=sc.s3.session_token,
                access_key=sc.s3.secret_access_key,
                credentials_provider=daft_s3_credentials,
                buffer_time=sc.s3.buffer_time,
                max_connections=sc.s3.max_connections,
                retry_initial_backoff_ms=sc.s3.retry_initial_backoff_ms,
                connect_timeout_ms=sc.s3.connect_timeout_ms,
                read_timeout_ms=sc.s3.read_timeout_ms,
                num_tries=sc.s3.num_tries,
                retry_mode=sc.s3.retry_mode,
                anonymous=sc.s3.anonymous,
                use_ssl=sc.s3.use_ssl,
                verify_ssl=sc.s3.verify_ssl,
                check_hostname_ssl=sc.s3.check_hostname_ssl,
                requester_pays=sc.s3.requester_pays,
                force_virtual_addressing=sc.s3.force_virtual_addressing,
                profile_name=sc.s3.profile_name,
            )

        if sc.gcs:
            daft_gcs_config = DaftGCSConfig(
                project_id=sc.gcs.project_id,
                credentials=sc.gcs.credentials,
                token=sc.gcs.token,
                anonymous=sc.gcs.anonymous,
            )
        if sc.azure:
            daft_azure_config = DaftAzureConfig(
                storage_account=sc.azure.storage_account,
                access_key=sc.azure.access_key,
                sas_token=sc.azure.sas_token,
                tenant_id=sc.azure.tenant_id,
                client_id=sc.azure.client_id,
                client_secret=sc.azure.client_secret,
                anonymous=sc.azure.anonymous,
            )

        if sc.local:
            daft_local_config = DaftLocalConfig(
                local_path=sc.local.local_path,
                anonymous=sc.local.anonymous,
            )

        storage_config = StorageConfig(
            s3=daft_s3_config,
            gcs=daft_gcs_config,
            azure=daft_azure_config,
            local=daft_local_config,
        )

        return storage_config
