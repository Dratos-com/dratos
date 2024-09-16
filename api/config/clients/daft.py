from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseSettings
from daft.io import (
    S3Config,
    S3Credentials,
    GCSConfig,
    AzureConfig,
)

from api.config.resources.storage_config import StorageConfig, LocalStorageConfig
from api.config.clients.base_context import BaseContext


# Daft IO Config
# https://www.getdaft.io/projects/docs/en/latest/api_docs/doc_gen/io_configs/daft.io.IOConfig.html


class DaftLocalConfig(LocalStorageConfig):

    def __init__(self, sc: StorageConfig):
        self.base_path=sc.local.base_path
        self.anonymous=sc.local.anonymous
    

class DaftS3Config(S3Config):

    def __init__(self, sc: StorageConfig):
        self.region_name=sc.s3.region_name
        self.endpoint_url=sc.s3.endpoint_url
        self.key_id=sc.s3.access_key_id
        self.session_token=sc.s3.session_token
        self.access_key=sc.s3.secret_access_key
        self.credentials_provider= S3Credentials(
            key_id=sc.s3.access_key_id,
            access_key=sc.s3.secret_access_key,
                session_token=sc.s3.session_token,
                expiry=sc.s3.expiry,
            )
        self.buffer_time=sc.s3.buffer_time
        self.max_connections=sc.s3.max_connections
        self.retry_initial_backoff_ms=sc.s3.retry_initial_backoff_ms
        self.connect_timeout_ms=sc.s3.connect_timeout_ms
        self.read_timeout_ms=sc.s3.read_timeout_ms
        self.num_tries=sc.s3.num_tries
        self.retry_mode=sc.s3.retry_mode
        self.anonymous=sc.s3.anonymous
        self.use_ssl=sc.s3.use_ssl
        self.verify_ssl=sc.s3.verify_ssl
        self.check_hostname_ssl=sc.s3.check_hostname_ssl
        self.requester_pays=sc.s3.requester_pays
        self.force_virtual_addressing=sc.s3.force_virtual_addressing
        self.profile_name=sc.s3.profile_name
        

class DaftGCSConfig(GCSConfig):

    def __init__(self, sc: StorageConfig):
        self.project_id=sc.gcs.project_id
        self.credentials=sc.gcs.credentials
        self.token=sc.gcs.token
        self.anonymous=sc.gcs.anonymous


class DaftAzureConfig(AzureConfig()):

    def __init__(self, sc: StorageConfig):
        self.account_name=sc.azure.account_name
        self.account_key=sc.azure.account_key
        self.anonymous=sc.azure.anonymous
        self.token=sc.azure.token
        self.endpoint=sc.azure.endpoint
        self.connection_string=sc.azure.connection_string
        self.sas_token=sc.azure.sas_token
        
class DaftExecutionPlanConfig(daft.context.):