Configuration
Setting the Runner
Control the execution backend that Daft will run on by calling these functions once at the start of your application.

daft.context.set_runner_py

Set the runner for executing Daft dataframes to your local Python interpreter - this is the default behavior.

daft.context.set_runner_ray

Set the runner for executing Daft dataframes to a Ray cluster

Setting configurations
Configure Daft in various ways during execution.

daft.set_planning_config

Globally sets various configuration parameters which control Daft plan construction behavior.

daft.planning_config_ctx

Context manager that wraps set_planning_config to reset the config to its original setting afternwards

daft.set_execution_config

Globally sets various configuration parameters which control various aspects of Daft execution.

daft.execution_config_ctx

Context manager that wraps set_execution_config to reset the config to its original setting afternwards

I/O Configurations
Configure behavior when Daft interacts with storage (e.g. credentials, retry policies and various other knobs to control performance/resource usage)

These configurations are most often used as inputs to Daft DataFrame reading I/O functions such as in Dataframe Creation.

daft.io.IOConfig

Create configurations to be used when accessing storage

daft.io.S3Config

Create configurations to be used when accessing an S3-compatible system

daft.io.S3Credentials

Create credentials to be used when accessing an S3-compatible system

daft.io.GCSConfig

Create configurations to be used when accessing Google Cloud Storage.

daft.io.AzureConfig

Create configurations to be used when accessing Azure Blob Storage.


daft.context.set_runner_py
daft.context.set_runner_py(use_thread_pool: Optional[bool] = None) → DaftContext[source]
Set the runner for executing Daft dataframes to your local Python interpreter - this is the default behavior.

Alternatively, users can set this behavior via an environment variable: DAFT_RUNNER=py

Returns
:
Daft context after setting the Py runner

Return type
:
DaftContext


daft.context.set_runner_ray
daft.context.set_runner_ray(address: Optional[str] = None, noop_if_initialized: bool = False, max_task_backlog: Optional[int] = None) → DaftContext[source]
Set the runner for executing Daft dataframes to a Ray cluster

Alternatively, users can set this behavior via environment variables:

DAFT_RUNNER=ray

Optionally, RAY_ADDRESS=ray://…

This function will throw an error if called multiple times in the same process.

Parameters
:
address – Address to head node of the Ray cluster. Defaults to None.

noop_if_initialized – If set to True, only the first call to this function will have any effect in setting the Runner. Subsequent calls will have no effect at all. Defaults to False, which throws an error if this function is called more than once per process.

Returns
:
Daft context after setting the Ray runner

Return type
:
DaftContext


daft.set_planning_config
daft.set_planning_config(config: Optional[PyDaftPlanningConfig] = None, default_io_config: Optional[IOConfig] = None) → DaftContext[source]
Globally sets various configuration parameters which control Daft plan construction behavior. These configuration values are used when a Dataframe is being constructed (e.g. calls to create a Dataframe, or to build on an existing Dataframe)

Parameters
:
config – A PyDaftPlanningConfig object to set the config to, before applying other kwargs. Defaults to None which indicates that the old (current) config should be used.

default_io_config – A default IOConfig to use in the absence of one being explicitly passed into any Expression (e.g. url.download()) or Dataframe operation (e.g. daft.read_parquet()).


daft.planning_config_ctx
daft.planning_config_ctx(**kwargs)[source]
Context manager that wraps set_planning_config to r

eset the config to its original setting afternwards


daft.set_execution_config
daft.set_execution_config(config: Optional[PyDaftExecutionConfig] = None, scan_tasks_min_size_bytes: Optional[int] = None, scan_tasks_max_size_bytes: Optional[int] = None, broadcast_join_size_bytes_threshold: Optional[int] = None, parquet_split_row_groups_max_files: Optional[int] = None, sort_merge_join_sort_with_aligned_boundaries: Optional[bool] = None, hash_join_partition_size_leniency: Optional[bool] = None, sample_size_for_sort: Optional[int] = None, num_preview_rows: Optional[int] = None, parquet_target_filesize: Optional[int] = None, parquet_target_row_group_size: Optional[int] = None, parquet_inflation_factor: Optional[float] = None, csv_target_filesize: Optional[int] = None, csv_inflation_factor: Optional[float] = None, shuffle_aggregation_default_partitions: Optional[int] = None, read_sql_partition_size_bytes: Optional[int] = None, enable_aqe: Optional[bool] = None, enable_native_executor: Optional[bool] = None, default_morsel_size: Optional[int] = None) → DaftContext[source]
Globally sets various configuration parameters which control various aspects of Daft execution. These configuration values are used when a Dataframe is executed (e.g. calls to write_*, collect() or show())

Parameters
:
config – A PyDaftExecutionConfig object to set the config to, before applying other kwargs. Defaults to None which indicates that the old (current) config should be used.

scan_tasks_min_size_bytes – Minimum size in bytes when merging ScanTasks when reading files from storage. Increasing this value will make Daft perform more merging of files into a single partition before yielding, which leads to bigger but fewer partitions. (Defaults to 96 MiB)

scan_tasks_max_size_bytes – Maximum size in bytes when merging ScanTasks when reading files from storage. Increasing this value will increase the upper bound of the size of merged ScanTasks, which leads to bigger but fewer partitions. (Defaults to 384 MiB)

broadcast_join_size_bytes_threshold – If one side of a join is smaller than this threshold, a broadcast join will be used. Default is 10 MiB.

parquet_split_row_groups_max_files – Maximum number of files to read in which the row group splitting should happen. (Defaults to 10)

sort_merge_join_sort_with_aligned_boundaries – Whether to use a specialized algorithm for sorting both sides of a sort-merge join such that they have aligned boundaries. This can lead to a faster merge-join at the cost of more skewed sorted join inputs, increasing the risk of OOMs.

hash_join_partition_size_leniency – If the left side of a hash join is already correctly partitioned and the right side isn’t, and the ratio between the left and right size is at least this value, then the right side is repartitioned to have an equal number of partitions as the left. Defaults to 0.5.

sample_size_for_sort – number of elements to sample from each partition when running sort, Default is 20.

num_preview_rows – number of rows to when showing a dataframe preview, Default is 8.

parquet_target_filesize – Target File Size when writing out Parquet Files. Defaults to 512MB

parquet_target_row_group_size – Target Row Group Size when writing out Parquet Files. Defaults to 128MB

parquet_inflation_factor – Inflation Factor of parquet files (In-Memory-Size / File-Size) ratio. Defaults to 3.0

csv_target_filesize – Target File Size when writing out CSV Files. Defaults to 512MB

csv_inflation_factor – Inflation Factor of CSV files (In-Memory-Size / File-Size) ratio. Defaults to 0.5

shuffle_aggregation_default_partitions – Minimum number of partitions to create when performing aggregations. Defaults to 200, unless the number of input partitions is less than 200.

read_sql_partition_size_bytes – Target size of partition when reading from SQL databases. Defaults to 512MB

enable_aqe – Enables Adaptive Query Execution, Defaults to False

enable_native_executor – Enables new local executor. Defaults to False

default_morsel_size – Default size of morsels used for the new local executor. Defaults to 131072 rows.


daft.execution_config_ctx
daft.execution_config_ctx(**kwargs)[source]
Context manager that wraps set_execution_config to reset the config to its original setting afternwards

daft.io.IOConfig
class daft.io.IOConfig(s3=None, azure=None, gcs=None, http=None)
Create configurations to be used when accessing storage

Parameters
:
s3 – Configuration to use when accessing URLs with the s3:// scheme

azure – Configuration to use when accessing URLs with the az:// or abfs:// scheme

gcs – Configuration to use when accessing URLs with the gs:// or gcs:// scheme

Example

io_config = IOConfig(s3=S3Config(key_id="xxx", access_key="xxx", num_tries=10), azure=AzureConfig(anonymous=True), gcs=GCSConfig(...))
daft.read_parquet(["s3://some-path", "az://some-other-path", "gs://path3"], io_config=io_config)
__init__()
Methods

__init__()

from_json(input)

replace([s3, azure, gcs, http])

Attributes

azure

Configuration to be used when accessing Azure URLs

gcs

Configuration to be used when accessing Azure URLs

http

Configuration to be used when accessing Azure URLs

s3

Configuration to be used when accessing s3 URLs

daft.io.S3Config
class daft.io.S3Config(region_name=None, endpoint_url=None, key_id=None, session_token=None, access_key=None, credentials_provider=None, buffer_time=None, max_connections=None, retry_initial_backoff_ms=None, connect_timeout_ms=None, read_timeout_ms=None, num_tries=None, retry_mode=None, anonymous=None, use_ssl=None, verify_ssl=None, check_hostname_ssl=None, requester_pays=None, force_virtual_addressing=None, profile_name=None)
Create configurations to be used when accessing an S3-compatible system

Parameters
:
region_name (str, optional) – Name of the region to be used (used when accessing AWS S3), defaults to “us-east-1”. If wrongly provided, Daft will attempt to auto-detect the buckets’ region at the cost of extra S3 requests.

endpoint_url (str, optional) – URL to the S3 endpoint, defaults to endpoints to AWS

key_id (str, optional) – AWS Access Key ID, defaults to auto-detection from the current environment

access_key (str, optional) – AWS Secret Access Key, defaults to auto-detection from the current environment

credentials_provider (Callable[[], S3Credentials], optional) – Custom credentials provider function, should return a S3Credentials object

buffer_time (int, optional) – Amount of time in seconds before the actual credential expiration time where credentials given by credentials_provider are considered expired, defaults to 10s

max_connections (int, optional) – Maximum number of connections to S3 at any time, defaults to 64

session_token (str, optional) – AWS Session Token, required only if key_id and access_key are temporary credentials

retry_initial_backoff_ms (int, optional) – Initial backoff duration in milliseconds for an S3 retry, defaults to 1000ms

connect_timeout_ms (int, optional) – Timeout duration to wait to make a connection to S3 in milliseconds, defaults to 10 seconds

read_timeout_ms (int, optional) – Timeout duration to wait to read the first byte from S3 in milliseconds, defaults to 10 seconds

num_tries (int, optional) – Number of attempts to make a connection, defaults to 5

retry_mode (str, optional) – Retry Mode when a request fails, current supported values are standard and adaptive, defaults to adaptive

anonymous (bool, optional) – Whether or not to use “anonymous mode”, which will access S3 without any credentials

use_ssl (bool, optional) – Whether or not to use SSL, which require accessing S3 over HTTPS rather than HTTP, defaults to True

verify_ssl (bool, optional) – Whether or not to verify ssl certificates, which will access S3 without checking if the certs are valid, defaults to True

check_hostname_ssl (bool, optional) – Whether or not to verify the hostname when verifying ssl certificates, this was the legacy behavior for openssl, defaults to True

requester_pays (bool, optional) – Whether or not the authenticated user will assume transfer costs, which is required by some providers of bulk data, defaults to False

force_virtual_addressing (bool, optional) – Force S3 client to use virtual addressing in all cases. If False, virtual addressing will only be used if endpoint_url is empty, defaults to False

profile_name (str, optional) – Name of AWS_PROFILE to load, defaults to None which will then check the Environment Variable AWS_PROFILE then fall back to default

Example

io_config = IOConfig(s3=S3Config(key_id="xxx", access_key="xxx"))
daft.read_parquet("s3://some-path", io_config=io_config)
__init__()
Methods

__init__()

from_env()

Creates an S3Config from the current environment, auto-discovering variables such as credentials, regions and more.

replace([region_name, endpoint_url, key_id, ...])

Attributes

access_key

AWS Secret Access Key

anonymous

AWS Anonymous Mode

buffer_time

AWS Buffer Time in Seconds

check_hostname_ssl

AWS Check SSL Hostname

connect_timeout_ms

AWS Connection Timeout in Milliseconds

credentials_provider

Custom credentials provider function

endpoint_url

S3-compatible endpoint to use

force_virtual_addressing

AWS force virtual addressing

key_id

AWS Access Key ID

max_connections

AWS max connections per IO thread

num_tries

AWS Number Retries

profile_name

AWS profile name

read_timeout_ms

AWS Read Timeout in Milliseconds

region_name

Region to use when accessing AWS S3

requester_pays

AWS Requester Pays

retry_initial_backoff_ms

AWS Retry Initial Backoff Time in Milliseconds

retry_mode

AWS Retry Mode

session_token

AWS Session Token

use_ssl

AWS Use SSL

verify_ssl

AWS Verify SSL



daft.io.S3Credentials
class daft.io.S3Credentials(key_id, access_key, session_token=None, expiry=None)
Create credentials to be used when accessing an S3-compatible system

Parameters
:
key_id (str) – AWS Access Key ID, defaults to auto-detection from the current environment

access_key (str) – AWS Secret Access Key, defaults to auto-detection from the current environment

session_token (str, optional) – AWS Session Token, required only if key_id and access_key are temporary credentials

expiry (datetime.datetime, optional) – Expiry time of the credentials, credentials are assumed to be permanent if not provided

Example

get_credentials = lambda: S3Credentials(
    key_id="xxx",
    access_key="xxx",
    expiry=(datetime.datetime.now() + datetime.timedelta(hours=1))
)
io_config = IOConfig(s3=S3Config(credentials_provider=get_credentials))
daft.read_parquet("s3://some-path", io_config=io_config)
__init__()
Methods

__init__()

Attributes

access_key

AWS Secret Access Key

expiry

AWS Session Token

key_id

AWS Access Key ID


daft.io.GCSConfig
class daft.io.GCSConfig(project_id=None, credentials=None, token=None, anonymous=None)
Create configurations to be used when accessing Google Cloud Storage. Credentials may be provided directly with the credentials parameter, or set with the GOOGLE_APPLICATION_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS environment variables.

Parameters
:
project_id (str, optional) – Google Project ID, defaults to value in credentials file or Google Cloud metadata service

credentials (str, optional) – Path to credentials file or JSON string with credentials

token (str, optional) – OAuth2 token to use for authentication. You likely want to use credentials instead, since it can be used to refresh the token. This value is used when vended by a data catalog.

anonymous (bool, optional) – Whether or not to use “anonymous mode”, which will access Google Storage without any credentials. Defaults to false

Example

io_config = IOConfig(gcs=GCSConfig(anonymous=True))
daft.read_parquet("gs://some-path", io_config=io_config)
__init__()
Methods

__init__()

replace([project_id, credentials, token, ...])

Attributes

anonymous

Whether to use anonymous mode

credentials

Credentials file path or string to use when accessing Google Cloud Storage

project_id

Project ID to use when accessing Google Cloud Storage

token

OAuth2 token to use when accessing Google Cloud Storage

daft.io.AzureConfig
class daft.io.AzureConfig(storage_account=None, access_key=None, sas_token=None, bearer_token=None, tenant_id=None, client_id=None, client_secret=None, use_fabric_endpoint=None, anonymous=None, endpoint_url=None, use_ssl=None)
Create configurations to be used when accessing Azure Blob Storage. To authenticate with Microsoft Entra ID, tenant_id, client_id, and client_secret must be provided. If no credentials are provided, Daft will attempt to fetch credentials from the environment.

Parameters
:
storage_account (str) – Azure Storage Account, defaults to reading from AZURE_STORAGE_ACCOUNT environment variable.

access_key (str, optional) – Azure Secret Access Key, defaults to reading from AZURE_STORAGE_KEY environment variable

sas_token (str, optional) – Shared Access Signature token, defaults to reading from AZURE_STORAGE_SAS_TOKEN environment variable

bearer_token (str, optional) – Bearer Token, defaults to reading from AZURE_STORAGE_TOKEN environment variable

tenant_id (str, optional) – Azure Tenant ID

client_id (str, optional) – Azure Client ID

client_secret (str, optional) – Azure Client Secret

use_fabric_endpoint (bool, optional) – Whether to use Microsoft Fabric, you may want to set this if your URLs are from “fabric.microsoft.com”. Defaults to false

anonymous (bool, optional) – Whether or not to use “anonymous mode”, which will access Azure without any credentials

endpoint_url (str, optional) – Custom URL to the Azure endpoint, e.g. https://my-account-name.blob.core.windows.net. Overrides use_fabric_endpoint if set

use_ssl (bool, optional) – Whether or not to use SSL, which require accessing Azure over HTTPS rather than HTTP, defaults to True

Example

io_config = IOConfig(azure=AzureConfig(storage_account="dafttestdata", access_key="xxx"))
daft.read_parquet("az://some-path", io_config=io_config)
__init__()
Methods

__init__()

replace([storage_account, access_key, ...])

Attributes

access_key

Azure Secret Access Key

anonymous

Whether access is anonymous

bearer_token

Azure Bearer Token

client_id

client_secret

endpoint_url

Azure Secret Access Key

sas_token

Azure Shared Access Signature token

storage_account

Storage Account to use when accessing Azure Storage

tenant_id

use_fabric_endpoint

Whether to use Microsoft Fabric

use_ssl

Whether SSL (HTTPS) is required





daft.context.set_runner_py
daft.context.set_runner_py(use_thread_pool: Optional[bool] = None) → DaftContext[source]
Set the runner for executing Daft dataframes to your local Python interpreter - this is the default behavior.

Alternatively, users can set this behavior via an environment variable: DAFT_RUNNER=py

Returns
:
Daft context after setting the Py runner

Return type
:
DaftContext


daft.context.set_runner_ray
daft.context.set_runner_ray(address: Optional[str] = None, noop_if_initialized: bool = False, max_task_backlog: Optional[int] = None) → DaftContext[source]
Set the runner for executing Daft dataframes to a Ray cluster

Alternatively, users can set this behavior via environment variables:

DAFT_RUNNER=ray

Optionally, RAY_ADDRESS=ray://…

This function will throw an error if called multiple times in the same process.

Parameters
:
address – Address to head node of the Ray cluster. Defaults to None.

noop_if_initialized – If set to True, only the first call to this function will have any effect in setting the Runner. Subsequent calls will have no effect at all. Defaults to False, which throws an error if this function is called more than once per process.

Returns
:
Daft context after setting the Ray runner

Return type
:
DaftContext



daft.set_planning_config
daft.set_planning_config(config: Optional[PyDaftPlanningConfig] = None, default_io_config: Optional[IOConfig] = None) → DaftContext[source]
Globally sets various configuration parameters which control Daft plan construction behavior. These configuration values are used when a Dataframe is being constructed (e.g. calls to create a Dataframe, or to build on an existing Dataframe)

Parameters
:
config – A PyDaftPlanningConfig object to set the config to, before applying other kwargs. Defaults to None which indicates that the old (current) config should be used.

default_io_config – A default IOConfig to use in the absence of one being explicitly passed into any Expression (e.g. url.download()) or Dataframe operation (e.g. daft.read_parquet()).


daft.planning_config_ctx
daft.planning_config_ctx(**kwargs)[source]
Context manager that wraps set_planning_config to reset the config to its original setting afternwards

daft.set_execution_config
daft.set_execution_config(config: Optional[PyDaftExecutionConfig] = None, scan_tasks_min_size_bytes: Optional[int] = None, scan_tasks_max_size_bytes: Optional[int] = None, broadcast_join_size_bytes_threshold: Optional[int] = None, parquet_split_row_groups_max_files: Optional[int] = None, sort_merge_join_sort_with_aligned_boundaries: Optional[bool] = None, hash_join_partition_size_leniency: Optional[bool] = None, sample_size_for_sort: Optional[int] = None, num_preview_rows: Optional[int] = None, parquet_target_filesize: Optional[int] = None, parquet_target_row_group_size: Optional[int] = None, parquet_inflation_factor: Optional[float] = None, csv_target_filesize: Optional[int] = None, csv_inflation_factor: Optional[float] = None, shuffle_aggregation_default_partitions: Optional[int] = None, read_sql_partition_size_bytes: Optional[int] = None, enable_aqe: Optional[bool] = None, enable_native_executor: Optional[bool] = None, default_morsel_size: Optional[int] = None) → DaftContext[source]
Globally sets various configuration parameters which control various aspects of Daft execution. These configuration values are used when a Dataframe is executed (e.g. calls to write_*, collect() or show())

Parameters
:
config – A PyDaftExecutionConfig object to set the config to, before applying other kwargs. Defaults to None which indicates that the old (current) config should be used.

scan_tasks_min_size_bytes – Minimum size in bytes when merging ScanTasks when reading files from storage. Increasing this value will make Daft perform more merging of files into a single partition before yielding, which leads to bigger but fewer partitions. (Defaults to 96 MiB)

scan_tasks_max_size_bytes – Maximum size in bytes when merging ScanTasks when reading files from storage. Increasing this value will increase the upper bound of the size of merged ScanTasks, which leads to bigger but fewer partitions. (Defaults to 384 MiB)

broadcast_join_size_bytes_threshold – If one side of a join is smaller than this threshold, a broadcast join will be used. Default is 10 MiB.

parquet_split_row_groups_max_files – Maximum number of files to read in which the row group splitting should happen. (Defaults to 10)

sort_merge_join_sort_with_aligned_boundaries – Whether to use a specialized algorithm for sorting both sides of a sort-merge join such that they have aligned boundaries. This can lead to a faster merge-join at the cost of more skewed sorted join inputs, increasing the risk of OOMs.

hash_join_partition_size_leniency – If the left side of a hash join is already correctly partitioned and the right side isn’t, and the ratio between the left and right size is at least this value, then the right side is repartitioned to have an equal number of partitions as the left. Defaults to 0.5.

sample_size_for_sort – number of elements to sample from each partition when running sort, Default is 20.

num_preview_rows – number of rows to when showing a dataframe preview, Default is 8.

parquet_target_filesize – Target File Size when writing out Parquet Files. Defaults to 512MB

parquet_target_row_group_size – Target Row Group Size when writing out Parquet Files. Defaults to 128MB

parquet_inflation_factor – Inflation Factor of parquet files (In-Memory-Size / File-Size) ratio. Defaults to 3.0

csv_target_filesize – Target File Size when writing out CSV Files. Defaults to 512MB

csv_inflation_factor – Inflation Factor of CSV files (In-Memory-Size / File-Size) ratio. Defaults to 0.5

shuffle_aggregation_default_partitions – Minimum number of partitions to create when performing aggregations. Defaults to 200, unless the number of input partitions is less than 200.

read_sql_partition_size_bytes – Target size of partition when reading from SQL databases. Defaults to 512MB

enable_aqe – Enables Adaptive Query Execution, Defaults to False

enable_native_executor – Enables new local executor. Defaults to False

default_morsel_size – Default size of morsels used for the new local executor. Defaults to 131072 rows.

daft.execution_config_ctx
daft.execution_config_ctx(**kwargs)[source]
Context manager that wraps set_execution_config to reset the config to its original setting afterwards