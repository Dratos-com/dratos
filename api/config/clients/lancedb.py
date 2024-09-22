from pydantic_settings import BaseSettings
from typing import List, Optional
from dynaconf import Dynaconf

from api.config.context.storage_context import StorageContext

# NOTE: LanceDB doesn't have config classes for each storage type. So we used defaults from the docs.
# https://lancedb.github.io/lancedb/guides/storage/

# fmt: off

class LanceDBBaseConfig:
    allow_http: bool = False                                      # Allow non-TLS (HTTP) connections
    allow_invalid_certificates: bool = False                      # Skip certificate validation on HTTPS connections
    connect_timeout: str = "5s"                                   # Timeout for the connect phase
    timeout: str = "30s"                                          # Timeout for the entire request
    user_agent: str = "lancedb-python/0.1.0"                      # User agent string
    proxy_url: Optional[str] = None                               # URL of the proxy server
    proxy_ca_certificate: Optional[str] = None                    # PEM-formatted CA certificate for proxy
    proxy_excludes: List[str] = []                                # Hosts that bypass the proxy


class LanceDBLocalConfig(LanceDBBaseConfig):
    local_path: Optional[str] = "~/lancedb/data"                  # Path for local storage
    anonymous: bool = False                                       # Enable anonymous access


class LanceDBS3Config(LanceDBBaseConfig):
    aws_endpoint: Optional[str] = None                            # S3 endpoint URL
    aws_access_key_id: Optional[str] = None                       # AWS Access Key ID
    aws_secret_access_key: Optional[str] = None                   # AWS Secret Access Key
    aws_session_token: Optional[str] = None                       # AWS Session Token
    aws_virtual_hosted_style: bool = False                        # Use virtual hosted-style URLs
    aws_s3_express: bool = False                                  # Use S3 Express One Zone endpoints
    aws_server_side_encryption: Optional[str] = None              # Server-side encryption algorithm
    aws_sse_kms_key_id: Optional[str] = None                      # KMS key ID for encryption
    aws_sse_bucket_key_enabled: bool = False                      # Enable bucket key encryption
    aws_region: Optional[str] = "us-west-2"                       # AWS region


class LanceDBGCSConfig(LanceDBBaseConfig):
    google_service_account: Optional[str] = None                   # Path to GCS service account JSON
    google_service_account_key: Optional[str] = None               # GCS service account key
    google_application_credentials: Optional[str] = None           # Path to application credentials


class LanceDBAzureConfig(LanceDBBaseConfig):
    azure_storage_account_name: Optional[str] = None               # Azure Storage Account Name
    azure_storage_account_key: Optional[str] = None                # Azure Storage Account Key
    azure_client_id: Optional[str] = None                          # Azure Client ID
    azure_client_secret: Optional[str] = None                      # Azure Client Secret
    azure_tenant_id: Optional[str] = None                          # Azure Tenant ID
    azure_storage_sas_key: Optional[str] = None                    # Azure SAS Token
    azure_storage_token: Optional[str] = None                      # Azure Storage Token
    azure_storage_use_emulator: bool = False                       # Use Azure Storage Emulator
    azure_endpoint: Optional[str] = None                           # Azure endpoint URL
    azure_use_fabric_endpoint: bool = False                        # Use Fabric endpoint
    azure_msi_endpoint: Optional[str] = None                       # MSI endpoint
    azure_object_id: Optional[str] = None                          # Object ID for MSI
    azure_msi_resource_id: Optional[str] = None                    # MSI Resource ID
    azure_federated_token_file: Optional[str] = None               # Federated Token File
    azure_use_azure_cli: bool = False                              # Use Azure CLI for authentication
    azure_disable_tagging: bool = False                            # Disable tagging.


class LanceDBConfig:
    storage_options: dict = {}                                     # Storage options dictionary

    def __init__(self, sc: StorageContext, settings: Dynaconf):

        lancedb_settings = settings.get('lancedb', {})
        
        # Initialize Local Storage Configuration
        if sc.local:
            self.local = LanceDBLocalConfig(
                allow_http=lancedb_settings.get('allow_http'),
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

        # Initialize S3 Storage Configuration
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

        # Initialize GCS Storage Configuration
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
                google_application_credentials=sc.gcs.google_application_credentials,
            )

        # Initialize Azure Storage Configuration
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
                azure_object_id=sc.azure.azure_object_id,
                azure_msi_resource_id=sc.azure.azure_msi_resource_id,
                azure_federated_token_file=sc.azure.azure_federated_token_file,
                azure_use_azure_cli=sc.azure.azure_use_azure_cli,
                azure_disable_tagging=sc.azure.azure_disable_tagging,
            )

# fmt: on
