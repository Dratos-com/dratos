import lancedb
from pydantic import BaseSettings
from typing import List, Optional
from dynaconf import Dynaconf

from api.config.resources.storage_config import StorageConfig
# LanceDB doesn't have config classes for each storage type. So we used defaults from the docs.
# https://lancedb.github.io/lancedb/guides/storage/


class LanceDBBaseConfig(BaseSettings):
    allow_http: bool = False
    allow_invalid_certificates: bool = False
    connect_timeout: str = "5s"
    timeout: str = "30s"
    user_agent: str = "lancedb-python/0.1.0"
    proxy_url: str = None
    proxy_ca_certificate: str = None
    proxy_excludes: List[str] = []


class LanceDBLocalConfig(LanceDBBaseConfig):
    local_path: str = None
    anonymous: bool = False


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


class LanceDBConfig:
    storage_options: dict = {}

    def __init__(self, sc: StorageConfig, settings: Dynaconf):
        if sc.local:
            self.local = LanceDBLocalConfig(
                allow_http=settings.lance.local.allow_http,
                allow_invalid_certificates=settings.lance.local.allow_invalid_certificates,
                connect_timeout=settings.lance.local.connect_timeout,
                timeout=settings.lance.local.timeout,
                user_agent=settings.lance.local.user_agent,
                proxy_url=settings.lance.local.proxy_url,
                proxy_ca_certificate=settings.lance.local.proxy_ca_certificate,
                proxy_excludes=settings.lance.local.proxy_excludes,
                local_path=sc.local.local_path,
                anonymous=sc.local.anonymous,
            )

        if sc.s3:
            self.s3 = LanceDBS3Config(
                allow_http=settings.lance.s3.allow_http,
                allow_invalid_certificates=settings.lance.s3.allow_invalid_certificates,
                connect_timeout=settings.lance.s3.connect_timeout,
                timeout=settings.lance.s3.timeout,
                user_agent=settings.lance.s3.user_agent,
                proxy_url=settings.lance.s3.proxy_url,
                proxy_ca_certificate=settings.lance.s3.proxy_ca_certificate,
                proxy_excludes=settings.lance.s3.proxy_excludes,
                aws_endpoint=sc.s3.aws_endpoint,
                aws_access_key_id=sc.s3.aws_access_key_id,
                aws_secret_access_key=sc.s3.aws_secret_access_key,
                aws_session_token=sc.s3.aws_session_token,
                aws_virtual_hosted_style=sc.s3.aws_virtual_hosted_style,
                aws_s3_express=sc.s3.aws_s3_express,
                aws_server_side_encryption=sc.s3.aws_server_side_encryption,
                aws_sse_kms_key_id=sc.s3.aws_sse_kms_key_id,
                aws_sse_bucket_key_enabled=sc.s3.aws_sse_bucket_key_enabled,
                aws_region=sc.s3.aws_region,
            )

        if sc.gcs:
            self.gcs = LanceDBGCSConfig(
                allow_http=settings.lance.gcs.allow_http,
                allow_invalid_certificates=settings.lance.gcs.allow_invalid_certificates,
                connect_timeout=settings.lance.gcs.connect_timeout,
                timeout=settings.lance.gcs.timeout,
                user_agent=settings.lance.gcs.user_agent,
                proxy_url=settings.lance.gcs.proxy_url,
                proxy_ca_certificate=settings.lance.gcs.proxy_ca_certificate,
                proxy_excludes=settings.lance.gcs.proxy_excludes,
                google_service_account=sc.gcs.google_service_account,
                google_service_account_key=sc.gcs.google_service_account_key,
            )

        if sc.azure:
            self.azure = LanceDBAzureConfig(
                allow_http=settings.lance.azure.allow_http,
                allow_invalid_certificates=settings.lance.azure.allow_invalid_certificates,
                connect_timeout=settings.lance.azure.connect_timeout,
                timeout=settings.lance.azure.timeout,
                user_agent=settings.lance.azure.user_agent,
                proxy_url=settings.lance.azure.proxy_url,
                proxy_ca_certificate=settings.lance.azure.proxy_ca_certificate,
                proxy_excludes=settings.lance.azure.proxy_excludes,
                azure_storage_account_name=sc.azure.azure_storage_account_name,
                azure_storage_account_key=sc.azure.azure_storage_account_key,
                azure_client_id=sc.azure.azure_client_id,
                azure_client_secret=sc.azure.azure_client_secret,
                azure_tenant_id=sc.azure.azure_tenant_id,
                azure_storage_sas_key=sc.azure.azure_storage_sas_key,
                azure_storage_token=sc.azure.azure_storage_token,
                azure_storage_use_emulator=sc.azure.azure_storage_use_emulator,
                azure_endpoint=sc.azure.azure_endpoint,
                azure_use_fabric_endpoint=sc.azure.azure_use_fabric_endpoint,
                azure_msi_endpoint=sc.azure.azure_msi_endpoint,
            )
