from typing import Optional
from datetime import datetime, timezone
from dynaconf import Dynaconf
from daft.daft import PyDaftExecutionConfig
from pydantic_settings import BaseSettings
from daft.io import (
    S3Config,
    S3Credentials,
    GCSConfig,
    AzureConfig,
)
from daft.execution.native_executor import NativeExecutor

from api.config.context.storage_context import StorageContext, LocalStorageContext


# Daft IO Config
# https://www.getdaft.io/projects/docs/en/latest/api_docs/doc_gen/io_configs/daft.io.IOConfig.html


# fmt: off

class DaftLocalConfig:
    def __init__(self, sc: StorageContext):
        self.config = PyDaftExecutionConfig().from_env()
        self.base_path = sc.local.local_path  # Base path for local storage
        self.anonymous = sc.local.anonymous  # Anonymous access flag


class DaftS3Config:
    def __init__(self, sc: StorageContext):
        self.config = S3Config(
            region_name=sc.s3.region_name,
            endpoint_url=sc.s3.endpoint_url,
            key_id=sc.s3.access_key_id,
            session_token=sc.s3.session_token,
            access_key=sc.s3.secret_access_key,
            credentials_provider=S3Credentials(
                key_id=sc.s3.access_key_id,
                access_key=sc.s3.secret_access_key,
                session_token=sc.s3.session_token,
                expiry=sc.s3.expiry,
            ),
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
            profile_name=sc.s3.profile_name
        )

    def __getattr__(self, name):
        return getattr(self.config, name)


class DaftGCSConfig:
    def __init__(self, sc: StorageContext):
        self.config = GCSConfig(
            project_id=sc.gcs.project_id,
            credentials=sc.gcs.credentials,
            token=sc.gcs.token,
            anonymous=sc.gcs.anonymous,
            google_service_account=sc.gcs.google_service_account,
            google_service_account_key=sc.gcs.google_service_account_key
        )
        self.project_id = sc.gcs.project_id                  # GCP Project ID
        self.credentials = sc.gcs.credentials                 # GCS credentials
        self.token = sc.gcs.token                       # OAuth2 token
        self.anonymous = sc.gcs.anonymous                   # Anonymous access
        self.google_service_account = sc.gcs.google_service_account      # GCS service account
        self.google_service_account_key = sc.gcs.google_service_account_key  # GCS service account key


class DaftAzureConfig:

    def __init__(self, sc: StorageContext):
        self.config = AzureConfig(
            account_name=sc.azure.account_name,
            account_key=sc.azure.account_key,
            anonymous=sc.azure.anonymous,
            token=sc.azure.token,
            endpoint=sc.azure.endpoint,
            connection_string=sc.azure.connection_string,
            sas_token=sc.azure.sas_token,
            storage_account_key=sc.azure.storage_account_key,
        )
        self.account_name= sc.azure.account_name                # Azure Storage Account Name
        self.account_key= sc.azure.account_key                 # Azure Access Key
        self.anonymous= sc.azure.anonymous                   # Anonymous access
        self.token= sc.azure.token                       # Bearer Token
        self.endpoint= sc.azure.endpoint                    # Azure endpoint URL
        self.connection_string= sc.azure.connection_string           # Azure connection string
        self.sas_token= sc.azure.sas_token                   # SAS token
        self.storage_account_key= sc.azure.storage_account_key         # Storage Account Key
        self.storage_sas_key= sc.azure.storage_sas_key             # Storage SAS Key
        self.storage_token= sc.azure.storage_token               # Storage Token
        self.use_fabric_endpoint= sc.azure.use_fabric_endpoint         # Use Fabric endpoint
        self.msi_endpoint= sc.azure.msi_endpoint                # MSI endpoint
        self.object_id= sc.azure.object_id                   # Object ID
        self.msi_resource_id= sc.azure.msi_resource_id             # MSI Resource ID
        self.federated_token_file= sc.azure.federated_token_file        # Federated Token File
        self.use_azure_cli= sc.azure.use_azure_cli               # Use Azure CLI
        self.disable_tagging= sc.azure.disable_tagging             # Disable tagging


class DaftExecutionPlanConfig:
    def __init__(self, sc: StorageContext, settings: Dynaconf):
        self.scan_tasks_min_size_bytes= settings.config.context.daft.execution.scan_tasks_min_size_bytes
        self.scan_tasks_max_size_bytes= settings.config.context.daft.execution.scan_tasks_max_size_bytes
        self.broadcast_join_size_bytes_threshold= settings.config.context.daft.execution.broadcast_join_size_bytes_threshold
        self.parquet_split_row_groups_max_files= settings.config.context.daft.execution.parquet_split_row_groups_max_files
        self.sort_merge_join_sort_with_aligned_boundaries= settings.config.context.daft.execution.sort_merge_join_sort_with_aligned_boundaries
        self.hash_join_partition_size_leniency= settings.config.context.daft.execution.hash_join_partition_size_leniency
        self.sample_size_for_sort= settings.config.context.daft.execution.sample_size_for_sort
        self.num_preview_rows= settings.config.context.daft.execution.num_preview_rows
        self.parquet_target_filesize= settings.config.context.daft.execution.parquet_target_filesize
        self.parquet_target_row_group_size= settings.config.context.daft.execution.parquet_target_row_group_size
        self.parquet_inflation_factor= settings.config.context.daft.execution.parquet_inflation_factor
        self.csv_target_filesize= settings.config.context.daft.execution.csv_target_filesize
        self.csv_inflation_factor= settings.config.context.daft.execution.csv_inflation_factor
        self.shuffle_aggregation_default_partitions= settings.config.context.daft.execution.shuffle_aggregation_default_partitions
        self.read_sql_partition_size_bytes= settings.config.context.daft.execution.read_sql_partition_size_bytes
        self.enable_aqe= settings.config.context.daft.execution.enable_aqe
        self.enable_native_executor= settings.config.context.daft.execution.enable_native_executor
        self.default_morsel_size= settings.config.context.daft.execution.default_morsel_size


# fmt: on
