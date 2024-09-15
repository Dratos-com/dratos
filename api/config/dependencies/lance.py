import lancedb

from typing import List
from api.config.resources.storage_config import StorageConfig
from api.config.resources.compute_config import ComputeConfig
from api.config.contexts.base_context import BaseContext, BaseLocalConfig

# LanceDB doesn't have config classes for each storage type. So we used defaults from the docs.
# https://lancedb.github.io/lancedb/guides/storage/


class LanceDBBaseConfig:
    allow_http: bool = False
    allow_invalid_certificates: bool = False
    connect_timeout: str = "5s"
    timeout: str = "30s"
    user_agent: str = "lancedb-python/0.1.0"
    proxy_url: str = None
    proxy_ca_certificate: str = None
    proxy_excludes: List[str] = []


class LanceDBLocalConfig(LanceDBBaseConfig, BaseLocalConfig):
    pass


class LanceDBS3Config(LanceDBBaseConfig):
    aws_endpoint: str = None  # The S3 endpoint to connect to.
    aws_access_key_id: str = None  # The AWS access key ID.
    aws_secret_access_key: str = None  # The AWS secret access key.
    aws_session_token: str = None  # The AWS session token.
    aws_virtual_hosted_style: bool = False  # Whether to use virtual hosted style URLs.
    aws_s3_express: bool = False  # Whether to use S3 Express.
    aws_server_side_encryption: bool = False  # Whether to use server-side encryption.
    aws_sse_kms_key_id: str = None  # The KMS key ID to use for server-side encryption.
    aws_sse_bucket_key_enabled: bool = False  # Whether to use bucket key encryption.
    aws_region: str = None  # The AWS region to connect to.


class LanceDBGCSConfig(LanceDBBaseConfig):
    google_service_account: str = None
    google_service_account_key: str = None
    google_application_credentials: str = None


class LanceDBAzureConfig(LanceDBBaseConfig):
    azure_storage_account_name: str = None
    azure_storage_account_key: str = None
    azure_client_id: str = None
    azure_client_secret: str = None
    azure_tenant_id: str = None
    azure_storage_sas_key: str = None
    azure_storage_token: str = None
    azure_storage_use_emulator: bool = False
    azure_endpoint: str = None
    azure_use_fabric_endpoint: bool = False
    azure_msi_endpoint: str = None
    azure_object_id: str = None
    azure_msi_resource_id: str = None
    azure_federated_token_file: str = None
    azure_use_azure_cli: bool = False
    azure_disable_tagging: bool = False


class LanceDBStorageConfig:
    s3: LanceDBS3Config = None
    gcs: LanceDBGCSConfig = None
    azure: LanceDBAzureConfig = None
    local: LanceDBLocalConfig = None


class LanceContext(BaseContext):
    def __init__(self, storage_context: StorageContext):
        def __init__(self, config_path: str = None):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        self.config_dict = config_dict["storage"]
        
        self.storage = self.configure_storage(storage_context)

    def configure_storage(self, sc: StorageContext) -> LanceDBStorageConfig:
        if sc.s3:
            lance_s3_config = LanceDBS3Config(
                allow_http=sc.lance.allow_http,
                allow_invalid_certificates=sc.lance.allow_invalid_certificates,
                connect_timeout=sc.lance.connect_timeout,
                timeout=sc.lance.timeout,
                user_agent=sc.lance.user_agent,
                proxy_url=sc.lance.proxy_url,
                proxy_ca_certificate=sc.lance.proxy_ca_certificate,
                proxy_excludes=sc.lance.proxy_excludes,
                aws_endpoint=sc.s3.endpoint_url,
                aws_access_key_id=sc.s3.access_key_id,
                aws_secret_access_key=sc.s3.secret_access_key,
                aws_session_token=sc.s3.session_token,
                aws_virtual_hosted_style=sc.s3.virtual_hosted_style,
                aws_s3_express=sc.s3.s3_express,
                aws_server_side_encryption=sc.s3.server_side_encryption,
                aws_sse_kms_key_id=sc.s3.sse_kms_key_id,
                aws_sse_bucket_key_enabled=sc.s3.sse_bucket_key_enabled,
                aws_region=sc.s3.region_name,
            )
        elif sc.gcs:
            lance_gcs_config = LanceDBGCSConfig(
                google_service_account=sc.gcs.service_account,
                google_service_account_key=sc.gcs.service_account_key,
                google_application_credentials=sc.gcs.application_credentials,
            )
        elif sc.azure:
            lance_azure_config = LanceDBAzureConfig(
                azure_storage_account_name=sc.azure.storage_account,
                azure_storage_account_key=sc.azure.storage_account_key,
                azure_client_id=sc.azure.client_id,
                azure_client_secret=sc.azure.client_secret,
                azure_tenant_id=sc.azure.tenant_id,
                azure_storage_sas_key=sc.azure.sas_key,
                azure_storage_token=sc.azure.token,
                azure_storage_use_emulator=sc.azure.use_emulator,
                azure_endpoint=sc.azure.endpoint,
                azure_use_fabric_endpoint=sc.azure.use_fabric_endpoint,
                azure_msi_endpoint=sc.azure.msi_endpoint,
                azure_object_id=sc.azure.object_id,
                azure_msi_resource_id=sc.azure.msi_resource_id,
                azure_federated_token_file=sc.azure.federated_token_file,
                azure_use_azure_cli=sc.azure.use_azure_cli,
                azure_disable_tagging=sc.azure.disable_tagging,
            )
        else:
            lance_local_config = LanceDBLocalConfig(
                local_path=sc.local.local_path,
                anonymous=sc.local.anonymous,
            )

        lance_config = LanceDBStorageConfig(
            s3=lance_s3_config,
            gcs=lance_gcs_config,
            azure=lance_azure_config,
            local=lance_local_config,
        )

        return lance_config

    def get_context(self):
        if self.storage:
            return self
        else:
            raise Exception("LanceDB storage not configured.")

    def set_context(self, storage_config: StorageConfig):
        self.storage = self.configure_storage(storage_config)
        return self

    def update_context(self, storage_config: StorageConfig):
        self.storage = self.configure_storage(storage_config)
        return self
