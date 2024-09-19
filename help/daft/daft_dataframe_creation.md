daft.from_pylist
daft.from_pylist(data: List[Dict[str, Any]]) → DataFrame[source]
Creates a DataFrame from a list of dictionaries.

Example

import daft
df = daft.from_pylist([{"foo": 1}, {"foo": 2}])
df.show()
╭───────╮
│ foo   │
│ ---   │
│ Int64 │
╞═══════╡
│ 1     │
├╌╌╌╌╌╌╌┤
│ 2     │
╰───────╯

(Showing first 2 of 2 rows)
Parameters
:
data – List of dictionaries, where each key is a column name.

Returns
:
DataFrame created from list of dictionaries.

Return type
:
DataFrame


daft.from_pydict
daft.from_pydict(data: Dict[str, Union[list, np.ndarray, pa.Array, pa.ChunkedArray]]) → DataFrame[source]
Creates a DataFrame from a Python dictionary.

Example

import daft
df = daft.from_pydict({"foo": [1, 2]})
df.show()
╭───────╮
│ foo   │
│ ---   │
│ Int64 │
╞═══════╡
│ 1     │
├╌╌╌╌╌╌╌┤
│ 2     │
╰───────╯

(Showing first 2 of 2 rows)
Parameters
:
data – Key -> Sequence[item] of data. Each Key is created as a column, and must have a value that is a Python list, Numpy array or PyArrow array. Values must be equal in length across all keys.

Returns
:
DataFrame created from dictionary of columns

Return type
:
DataFrame

daft.from_arrow
daft.from_arrow(data: Union[pa.Table, List[pa.Table], Iterable[pa.Table]]) → DataFrame[source]
Creates a DataFrame from a pyarrow Table.

Example

import pyarrow as pa
import daft
t = pa.table({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
df = daft.from_arrow(t)
df.show()
╭───────┬──────╮
│ a     ┆ b    │
│ ---   ┆ ---  │
│ Int64 ┆ Utf8 │
╞═══════╪══════╡
│ 1     ┆ foo  │
├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
│ 2     ┆ bar  │
├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
│ 3     ┆ baz  │
╰───────┴──────╯

(Showing first 3 of 3 rows)
Parameters
:
data – pyarrow Table(s) that we wish to convert into a Daft DataFrame.

Returns
:
DataFrame created from the provided pyarrow Table.

Return type
:
DataFrame

daft.from_pandas
daft.from_pandas(data: Union[pd.DataFrame, List[pd.DataFrame]]) → DataFrame[source]
Creates a Daft DataFrame from a pandas DataFrame.

Example

import pandas as pd
import daft
pd_df = pd.DataFrame({"a": [1, 2, 3], "b": ["foo", "bar", "baz"]})
df = daft.from_pandas(pd_df)
df.show()
╭───────┬──────╮
│ a     ┆ b    │
│ ---   ┆ ---  │
│ Int64 ┆ Utf8 │
╞═══════╪══════╡
│ 1     ┆ foo  │
├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
│ 2     ┆ bar  │
├╌╌╌╌╌╌╌┼╌╌╌╌╌╌┤
│ 3     ┆ baz  │
╰───────┴──────╯

(Showing first 3 of 3 rows)
Parameters
:
data – pandas DataFrame(s) that we wish to convert into a Daft DataFrame.

Returns
:
Daft DataFrame created from the provided pandas DataFrame.

Return type
:
DataFrame

daft.read_parquet
daft.read_parquet(path: Union[str, List[str]], row_groups: Optional[List[List[int]]] = None, infer_schema: bool = True, schema: Optional[Dict[str, DataType]] = None, io_config: Optional[IOConfig] = None, use_native_downloader: bool = True, coerce_int96_timestamp_unit: Optional[Union[str, TimeUnit]] = None, schema_hints: Optional[Dict[str, DataType]] = None, _multithreaded_io: Optional[bool] = None, _chunk_size: Optional[int] = None) → DataFrame[source]
Creates a DataFrame from Parquet file(s)

Example

df = daft.read_parquet("/path/to/file.parquet")
df = daft.read_parquet("/path/to/directory")
df = daft.read_parquet("/path/to/files-*.parquet")
df = daft.read_parquet("s3://path/to/files-*.parquet")
df = daft.read_parquet("gs://path/to/files-*.parquet")
Parameters
:
path (str) – Path to Parquet file (allows for wildcards)

row_groups (List[int] or List[List[int]]) – List of row groups to read corresponding to each file.

infer_schema (bool) – Whether to infer the schema of the Parquet, defaults to True.

schema (dict[str, DataType]) – A schema that is used as the definitive schema for the Parquet file if infer_schema is False, otherwise it is used as a schema hint that is applied after the schema is inferred.

io_config (IOConfig) – Config to be used with the native downloader

use_native_downloader – Whether to use the native downloader instead of PyArrow for reading Parquet.

coerce_int96_timestamp_unit – TimeUnit to coerce Int96 TimeStamps to. e.g.: [ns, us, ms], Defaults to None.

_multithreaded_io – Whether to use multithreading for IO threads. Setting this to False can be helpful in reducing the amount of system resources (number of connections and thread contention) when running in the Ray runner. Defaults to None, which will let Daft decide based on the runner it is currently using.

Returns
:
parsed DataFrame

Return type
:
DataFrame

daft.read_csv
daft.read_csv(path: Union[str, List[str]], infer_schema: bool = True, schema: Optional[Dict[str, DataType]] = None, has_headers: bool = True, delimiter: Optional[str] = None, double_quote: bool = True, quote: Optional[str] = None, escape_char: Optional[str] = None, comment: Optional[str] = None, allow_variable_columns: bool = False, io_config: Optional[IOConfig] = None, use_native_downloader: bool = True, schema_hints: Optional[Dict[str, DataType]] = None, _buffer_size: Optional[int] = None, _chunk_size: Optional[int] = None) → DataFrame[source]
Creates a DataFrame from CSV file(s)

Example

df = daft.read_csv("/path/to/file.csv")
df = daft.read_csv("/path/to/directory")
df = daft.read_csv("/path/to/files-*.csv")
df = daft.read_csv("s3://path/to/files-*.csv")
Parameters
:
path (str) – Path to CSV (allows for wildcards)

infer_schema (bool) – Whether to infer the schema of the CSV, defaults to True.

schema (dict[str, DataType]) – A schema that is used as the definitive schema for the CSV if infer_schema is False, otherwise it is used as a schema hint that is applied after the schema is inferred.

has_headers (bool) – Whether the CSV has a header or not, defaults to True

delimiter (Str) – Delimiter used in the CSV, defaults to “,”

doubled_quote (bool) – Whether to support double quote escapes, defaults to True

escape_char (str) – Character to use as the escape character for double quotes, or defaults to "

comment (str) – Character to treat as the start of a comment line, or None to not support comments

allow_variable_columns (bool) – Whether to allow for variable number of columns in the CSV, defaults to False. If set to True, Daft will append nulls to rows with less columns than the schema, and ignore extra columns in rows with more columns

io_config (IOConfig) – Config to be used with the native downloader

use_native_downloader – Whether to use the native downloader instead of PyArrow for reading Parquet. This is currently experimental.

Returns
:
parsed DataFrame

Return type
:
DataFrame


daft.read_json
daft.read_json(path: Union[str, List[str]], infer_schema: bool = True, schema: Optional[Dict[str, DataType]] = None, io_config: Optional[IOConfig] = None, use_native_downloader: bool = True, schema_hints: Optional[Dict[str, DataType]] = None, _buffer_size: Optional[int] = None, _chunk_size: Optional[int] = None) → DataFrame[source]
Creates a DataFrame from line-delimited JSON file(s)

Example

df = daft.read_json("/path/to/file.json")
df = daft.read_json("/path/to/directory")
df = daft.read_json("/path/to/files-*.json")
df = daft.read_json("s3://path/to/files-*.json")
Parameters
:
path (str) – Path to JSON files (allows for wildcards)

infer_schema (bool) – Whether to infer the schema of the JSON, defaults to True.

schema (dict[str, DataType]) – A schema that is used as the definitive schema for the JSON if infer_schema is False, otherwise it is used as a schema hint that is applied after the schema is inferred.

io_config (IOConfig) – Config to be used with the native downloader

use_native_downloader – Whether to use the native downloader instead of PyArrow for reading Parquet. This is currently experimental.

Returns
:
parsed DataFrame

Return type
:
DataFrame

daft.from_glob_path
daft.from_glob_path(path: str, io_config: Optional[IOConfig] = None) → DataFrame[source]
Creates a DataFrame of file paths and other metadata from a glob path.

This method supports wildcards:

“*” matches any number of any characters including none

“?” matches any single character

“[…]” matches any single character in the brackets

“**” recursively matches any number of layers of directories

The returned DataFrame will have the following columns:

path: the path to the file/directory

size: size of the object in bytes

type: either “file” or “directory”

Example

df = daft.from_glob_path("/path/to/files/*.jpeg")
df = daft.from_glob_path("/path/to/files/**/*.jpeg")
df = daft.from_glob_path("/path/to/files/**/image-?.jpeg")
Parameters
:
path (str) – Path to files on disk (allows wildcards).

io_config (IOConfig) – Configuration to use when running IO with remote services

Returns
:
DataFrame containing the path to each file as a row, along with other metadata
parsed from the provided filesystem.

Return type
:
DataFrame

daft.read_iceberg
daft.read_iceberg(pyiceberg_table: PyIcebergTable, snapshot_id: Optional[int] = None, io_config: Optional[IOConfig] = None) → DataFrame[source]
Create a DataFrame from an Iceberg table

Example

import pyiceberg

pyiceberg_table = pyiceberg.Table(...)
df = daft.read_iceberg(pyiceberg_table)

# Filters on this dataframe can now be pushed into
# the read operation from Iceberg
df = df.where(df["foo"] > 5)
df.show()
Note

This function requires the use of PyIceberg, which is the Apache Iceberg’s official project for Python.

Parameters
:
pyiceberg_table – Iceberg table created using the PyIceberg library

snapshot_id – Snapshot ID of the table to query

io_config – A custom IOConfig to use when accessing Iceberg object storage data. Defaults to None.

Returns
:
a DataFrame with the schema converted from the specified Iceberg table

Return type
:
DataFrame

daft.read_deltalake
daft.read_deltalake(table: Union[str, DataCatalogTable, UnityCatalogTable], io_config: Optional[IOConfig] = None, _multithreaded_io: Optional[bool] = None) → DataFrame[source]
Create a DataFrame from a Delta Lake table.

Example

df = daft.read_deltalake("some-table-uri")

# Filters on this dataframe can now be pushed into
# the read operation from Delta Lake.
df = df.where(df["foo"] > 5)
df.show()
Note

This function requires the use of deltalake, a Python library for interacting with Delta Lake.

Parameters
:
table – Either a URI for the Delta Lake table or a DataCatalogTable instance referencing a table in a data catalog, such as AWS Glue Data Catalog or Databricks Unity Catalog.

io_config – A custom IOConfig to use when accessing Delta Lake object storage data. Defaults to None.

_multithreaded_io – Whether to use multithreading for IO threads. Setting this to False can be helpful in reducing the amount of system resources (number of connections and thread contention) when running in the Ray runner. Defaults to None, which will let Daft decide based on the runner it is currently using.

Returns
:
A DataFrame with the schema converted from the specified Delta Lake table.

Return type
:
DataFrame

daft.read_hudi
daft.read_hudi(table_uri: str, io_config: Optional[IOConfig] = None) → DataFrame[source]
Create a DataFrame from a Hudi table.

Example

df = daft.read_hudi("some-table-uri")
df = df.where(df["foo"] > 5)
df.show()
Parameters
:
table_uri – URI to the Hudi table.

io_config – A custom IOConfig to use when accessing Hudi table object storage data. Defaults to None.

Returns
:
A DataFrame with the schema converted from the specified Hudi table.

Return type
:
DataFrame

daft.from_ray_dataset
daft.from_ray_dataset(ds: RayDataset) → DataFrame[source]
Creates a DataFrame from a Ray Dataset.

Note

This function can only work if Daft is running using the RayRunner.

Parameters
:
ds – The Ray Dataset to create a Daft DataFrame from.


daft.from_dask_dataframe
daft.from_dask_dataframe(ddf: dask.DataFrame) → DataFrame[source]
Creates a Daft DataFrame from a Dask DataFrame.

The provided Dask DataFrame must have been created using Dask-on-Ray.

Note

This function can only work if Daft is running using the RayRunner

Parameters
:
ddf – The Dask DataFrame to create a Daft DataFrame from.


daft.read_sql
daft.read_sql(sql: str, conn: Union[Callable[[], Connection], str], partition_col: Optional[str] = None, num_partitions: Optional[int] = None, disable_pushdowns_to_sql: bool = False, infer_schema: bool = True, infer_schema_length: int = 10, schema: Optional[Dict[str, DataType]] = None) → DataFrame[source]
Create a DataFrame from the results of a SQL query.

Parameters
:
sql (str) – SQL query to execute

conn (Union[Callable[[], Connection], str]) – SQLAlchemy connection factory or database URL

partition_col (Optional[str]) – Column to partition the data by, defaults to None

num_partitions (Optional[int]) – Number of partitions to read the data into, defaults to None, which will lets Daft determine the number of partitions.

disable_pushdowns_to_sql (bool) – Whether to disable pushdowns to the SQL query, defaults to False

infer_schema (bool) – Whether to turn on schema inference, defaults to True. If set to False, the schema parameter must be provided.

infer_schema_length (int) – The number of rows to scan when inferring the schema, defaults to 10. If infer_schema is False, this parameter is ignored. Note that if Daft is able to use ConnectorX to infer the schema, this parameter is ignored as ConnectorX is an Arrow backed driver.

schema (Optional[Dict[str, DataType]]) – A mapping of column names to datatypes. If infer_schema is False, this schema is used as the definitive schema for the data, otherwise it is used as a schema hint that is applied after the schema is inferred. This can be useful if the types can be more precisely determined than what the inference can provide (e.g., if a column can be declared as a fixed-sized list rather than a list).

Returns
:
Dataframe containing the results of the query

Return type
:
DataFrame

Note

Supported dialects:
Daft uses SQLGlot to build and translate SQL queries between dialects. For a list of supported dialects, see SQLGlot’s dialect documentation.

Partitioning:
When partition_col is specified, the function partitions the query based on that column. You can define num_partitions or leave it to Daft to decide. Daft calculates the specified column’s percentiles to determine partitions (e.g., for num_partitions=3, it uses the 33rd and 66th percentiles). If the database or column type lacks percentile calculation support, Daft partitions the query using equal ranges between the column’s minimum and maximum values.

Execution:
Daft executes SQL queries using using ConnectorX or SQLAlchemy, preferring ConnectorX unless a SQLAlchemy connection factory is specified or the database dialect is unsupported by ConnectorX.

Pushdowns:
Daft pushes down operations such as filtering, projections, and limits into the SQL query when possible. You can disable pushdowns by setting disable_pushdowns_to_sql=True, which will execute the SQL query as is.

Example

Read data from a SQL query and a database URL:

df = daft.read_sql("SELECT * FROM my_table", "sqlite:///my_database.db")
Read data from a SQL query and a SQLAlchemy connection factory:

def create_conn():
    return sqlalchemy.create_engine("sqlite:///my_database.db").connect()
df = daft.read_sql("SELECT * FROM my_table", create_conn)
Read data from a SQL query and partition the data by a column:

df = daft.read_sql(
    "SELECT * FROM my_table",
    "sqlite:///my_database.db",
    partition_col="id"
)
Read data from a SQL query and partition the data into 3 partitions:

df = daft.read_sql(
    "SELECT * FROM my_table",
    "sqlite:///my_database.db",
    partition_col="id",
    num_partitions=3
)

daft.read_lance
daft.read_lance(url: str, io_config: Optional[IOConfig] = None) → DataFrame[source]
Create a DataFrame from a LanceDB table

Note

This function requires the use of LanceDB, which is the Python library for the LanceDB project.

To ensure that this is installed with Daft, you may install: pip install getdaft[lance]

Example:

df = daft.read_lance("s3://my-lancedb-bucket/data/")
df.show()
Parameters
:
url – URL to the LanceDB table (supports remote URLs to object stores such as s3:// or gs://)

io_config – A custom IOConfig to use when accessing LanceDB data. Defaults to None.

Returns
:
a DataFrame with the schema converted from the specified LanceDB table

Return type
:
DataFrame