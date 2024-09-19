Python API Reference
This section contains the API reference for the OSS Python API.

Installation

pip install lancedb
The following methods describe the synchronous API client. There is also an asynchronous API client.

Connections (Synchronous)
lancedb.connect(uri: URI, *, api_key: Optional[str] = None, region: str = 'us-east-1', host_override: Optional[str] = None, read_consistency_interval: Optional[timedelta] = None, request_thread_pool: Optional[Union[int, ThreadPoolExecutor]] = None, **kwargs: Any) -> DBConnection
Connect to a LanceDB database.

Parameters:

Name	Type	Description	Default
uri	URI	The uri of the database.	required
api_key	Optional[str]	If presented, connect to LanceDB cloud. Otherwise, connect to a database on file system or cloud storage. Can be set via environment variable LANCEDB_API_KEY.	None
region	str	The region to use for LanceDB Cloud.	'us-east-1'
host_override	Optional[str]	The override url for LanceDB Cloud.	None
read_consistency_interval	Optional[timedelta]	(For LanceDB OSS only) The interval at which to check for updates to the table from other processes. If None, then consistency is not checked. For performance reasons, this is the default. For strong consistency, set this to zero seconds. Then every read will check for updates from other processes. As a compromise, you can set this to a non-zero timedelta for eventual consistency. If more than that interval has passed since the last check, then the table will be checked for updates. Note: this consistency only applies to read operations. Write operations are always consistent.	None
request_thread_pool	Optional[Union[int, ThreadPoolExecutor]]	The thread pool to use for making batch requests to the LanceDB Cloud API. If an integer, then a ThreadPoolExecutor will be created with that number of threads. If None, then a ThreadPoolExecutor will be created with the default number of threads. If a ThreadPoolExecutor, then that executor will be used for making requests. This is for LanceDB Cloud only and is only used when making batch requests (i.e., passing in multiple queries to the search method at once).	None
Examples:

For a local directory, provide a path for the database:


>>> import lancedb
>>> db = lancedb.connect("~/.lancedb")
For object storage, use a URI prefix:


>>> db = lancedb.connect("s3://my-bucket/lancedb")
Connect to LanceDB cloud:


>>> db = lancedb.connect("db://my_database", api_key="ldb_...")
Returns:

Name	Type	Description
conn	DBConnection	A connection to a LanceDB database.
Source code in lancedb/__init__.py
lancedb.db.DBConnection
Bases: EnforceOverrides

An active LanceDB connection interface.

Source code in lancedb/db.py
table_names(page_token: Optional[str] = None, limit: int = 10) -> Iterable[str] abstractmethod
List all tables in this database, in sorted order

Parameters:

Name	Type	Description	Default
page_token	Optional[str]	The token to use for pagination. If not present, start from the beginning. Typically, this token is last table name from the previous page. Only supported by LanceDb Cloud.	None
limit	int	The size of the page to return. Only supported by LanceDb Cloud.	10
Returns:

Type	Description
Iterable of str	
Source code in lancedb/db.py
create_table(name: str, data: Optional[DATA] = None, schema: Optional[Union[pa.Schema, LanceModel]] = None, mode: str = 'create', exist_ok: bool = False, on_bad_vectors: str = 'error', fill_value: float = 0.0, embedding_functions: Optional[List[EmbeddingFunctionConfig]] = None) -> Table abstractmethod
Create a Table in the database.

Parameters:

Name	Type	Description	Default
name	str	The name of the table.	required
data	Optional[DATA]	User must provide at least one of data or schema. Acceptable types are:
dict or list-of-dict

pandas.DataFrame

pyarrow.Table or pyarrow.RecordBatch

None
schema	Optional[Union[Schema, LanceModel]]	Acceptable types are:
pyarrow.Schema

LanceModel

None
mode	str	The mode to use when creating the table. Can be either "create" or "overwrite". By default, if the table already exists, an exception is raised. If you want to overwrite the table, use mode="overwrite".	'create'
exist_ok	bool	If a table by the same name already exists, then raise an exception if exist_ok=False. If exist_ok=True, then open the existing table; it will not add the provided data but will validate against any schema that's specified.	False
on_bad_vectors	str	What to do if any of the vectors are not the same size or contains NaNs. One of "error", "drop", "fill".	'error'
fill_value	float	The value to use when filling vectors. Only used if on_bad_vectors="fill".	0.0
Returns:

Type	Description
LanceTable	A reference to the newly created table.
!!! note	The vector index won't be created by default. To create the index, call the create_index method on the table.
Examples:

Can create with list of tuples or dictionaries:


>>> import lancedb
>>> db = lancedb.connect("./.lancedb")
>>> data = [{"vector": [1.1, 1.2], "lat": 45.5, "long": -122.7},
...         {"vector": [0.2, 1.8], "lat": 40.1, "long":  -74.1}]
>>> db.create_table("my_table", data)
LanceTable(connection=..., name="my_table")
>>> db["my_table"].head()
pyarrow.Table
vector: fixed_size_list<item: float>[2]
  child 0, item: float
lat: double
long: double
----
vector: [[[1.1,1.2],[0.2,1.8]]]
lat: [[45.5,40.1]]
long: [[-122.7,-74.1]]
You can also pass a pandas DataFrame:


>>> import pandas as pd
>>> data = pd.DataFrame({
...    "vector": [[1.1, 1.2], [0.2, 1.8]],
...    "lat": [45.5, 40.1],
...    "long": [-122.7, -74.1]
... })
>>> db.create_table("table2", data)
LanceTable(connection=..., name="table2")
>>> db["table2"].head()
pyarrow.Table
vector: fixed_size_list<item: float>[2]
  child 0, item: float
lat: double
long: double
----
vector: [[[1.1,1.2],[0.2,1.8]]]
lat: [[45.5,40.1]]
long: [[-122.7,-74.1]]
Data is converted to Arrow before being written to disk. For maximum control over how data is saved, either provide the PyArrow schema to convert to or else provide a PyArrow Table directly.


>>> custom_schema = pa.schema([
...   pa.field("vector", pa.list_(pa.float32(), 2)),
...   pa.field("lat", pa.float32()),
...   pa.field("long", pa.float32())
... ])
>>> db.create_table("table3", data, schema = custom_schema)
LanceTable(connection=..., name="table3")
>>> db["table3"].head()
pyarrow.Table
vector: fixed_size_list<item: float>[2]
  child 0, item: float
lat: float
long: float
----
vector: [[[1.1,1.2],[0.2,1.8]]]
lat: [[45.5,40.1]]
long: [[-122.7,-74.1]]
It is also possible to create an table from [Iterable[pa.RecordBatch]]:


>>> import pyarrow as pa
>>> def make_batches():
...     for i in range(5):
...         yield pa.RecordBatch.from_arrays(
...             [
...                 pa.array([[3.1, 4.1], [5.9, 26.5]],
...                     pa.list_(pa.float32(), 2)),
...                 pa.array(["foo", "bar"]),
...                 pa.array([10.0, 20.0]),
...             ],
...             ["vector", "item", "price"],
...         )
>>> schema=pa.schema([
...     pa.field("vector", pa.list_(pa.float32(), 2)),
...     pa.field("item", pa.utf8()),
...     pa.field("price", pa.float32()),
... ])
>>> db.create_table("table4", make_batches(), schema=schema)
LanceTable(connection=..., name="table4")
Source code in lancedb/db.py
open_table(name: str, *, index_cache_size: Optional[int] = None) -> Table
Open a Lance Table in the database.

Parameters:

Name	Type	Description	Default
name	str	The name of the table.	required
index_cache_size	Optional[int]	Set the size of the index cache, specified as a number of entries
The exact meaning of an "entry" will depend on the type of index: * IVF - there is one entry for each IVF partition * BTREE - there is one entry for the entire index

This cache applies to the entire opened table, across all indices. Setting this value higher will increase performance on larger datasets at the expense of more RAM

None
Returns:

Type	Description
A LanceTable object representing the table.	
Source code in lancedb/db.py
drop_table(name: str)
Drop a table from the database.

Parameters:

Name	Type	Description	Default
name	str	The name of the table.	required
Source code in lancedb/db.py
rename_table(cur_name: str, new_name: str)
Rename a table in the database.

Parameters:

Name	Type	Description	Default
cur_name	str	The current name of the table.	required
new_name	str	The new name of the table.	required
Source code in lancedb/db.py
drop_database()
Drop database This is the same thing as dropping all the tables

Source code in lancedb/db.py
Tables (Synchronous)
lancedb.table.Table
Bases: ABC

A Table is a collection of Records in a LanceDB Database.

Examples:

Create using DBConnection.create_table (more examples in that method's documentation).


>>> import lancedb
>>> db = lancedb.connect("./.lancedb")
>>> table = db.create_table("my_table", data=[{"vector": [1.1, 1.2], "b": 2}])
>>> table.head()
pyarrow.Table
vector: fixed_size_list<item: float>[2]
  child 0, item: float
b: int64
----
vector: [[[1.1,1.2]]]
b: [[2]]
Can append new data with Table.add().


>>> table.add([{"vector": [0.5, 1.3], "b": 4}])
Can query the table with Table.search.


>>> table.search([0.4, 0.4]).select(["b", "vector"]).to_pandas()
   b      vector  _distance
0  4  [0.5, 1.3]       0.82
1  2  [1.1, 1.2]       1.13
Search queries are much faster when an index is created. See Table.create_index.

Source code in lancedb/table.py

class Table(ABC):
    """
    A Table is a collection of Records in a LanceDB Database.

    Examples
    --------

    Create using [DBConnection.create_table][lancedb.DBConnection.create_table]
    (more examples in that method's documentation).

    >>> import lancedb
    >>> db = lancedb.connect("./.lancedb")
    >>> table = db.create_table("my_table", data=[{"vector": [1.1, 1.2], "b": 2}])
    >>> table.head()
    pyarrow.Table
    vector: fixed_size_list<item: float>[2]
      child 0, item: float
    b: int64
    ----
    vector: [[[1.1,1.2]]]
    b: [[2]]

    Can append new data with [Table.add()][lancedb.table.Table.add].

    >>> table.add([{"vector": [0.5, 1.3], "b": 4}])

    Can query the table with [Table.search][lancedb.table.Table.search].

    >>> table.search([0.4, 0.4]).select(["b", "vector"]).to_pandas()
       b      vector  _distance
    0  4  [0.5, 1.3]       0.82
    1  2  [1.1, 1.2]       1.13

    Search queries are much faster when an index is created. See
    [Table.create_index][lancedb.table.Table.create_index].
    """

    @property
    @abstractmethod
    def schema(self) -> pa.Schema:
        """The [Arrow Schema](https://arrow.apache.org/docs/python/api/datatypes.html#)
        of this Table

        """
        raise NotImplementedError

    @abstractmethod
    def count_rows(self, filter: Optional[str] = None) -> int:
        """
        Count the number of rows in the table.

        Parameters
        ----------
        filter: str, optional
            A SQL where clause to filter the rows to count.
        """
        raise NotImplementedError

    def to_pandas(self) -> "pd.DataFrame":
        """Return the table as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
        """
        return self.to_arrow().to_pandas()

    @abstractmethod
    def to_arrow(self) -> pa.Table:
        """Return the table as a pyarrow Table.

        Returns
        -------
        pa.Table
        """
        raise NotImplementedError

    def create_index(
        self,
        metric="L2",
        num_partitions=256,
        num_sub_vectors=96,
        vector_column_name: str = VECTOR_COLUMN_NAME,
        replace: bool = True,
        accelerator: Optional[str] = None,
        index_cache_size: Optional[int] = None,
    ):
        """Create an index on the table.

        Parameters
        ----------
        metric: str, default "L2"
            The distance metric to use when creating the index.
            Valid values are "L2", "cosine", or "dot".
            L2 is euclidean distance.
        num_partitions: int, default 256
            The number of IVF partitions to use when creating the index.
            Default is 256.
        num_sub_vectors: int, default 96
            The number of PQ sub-vectors to use when creating the index.
            Default is 96.
        vector_column_name: str, default "vector"
            The vector column name to create the index.
        replace: bool, default True
            - If True, replace the existing index if it exists.

            - If False, raise an error if duplicate index exists.
        accelerator: str, default None
            If set, use the given accelerator to create the index.
            Only support "cuda" for now.
        index_cache_size : int, optional
            The size of the index cache in number of entries. Default value is 256.
        """
        raise NotImplementedError

    @abstractmethod
    def create_scalar_index(
        self,
        column: str,
        *,
        replace: bool = True,
        index_type: Literal["BTREE", "BITMAP", "LABEL_LIST"] = "BTREE",
    ):
        """Create a scalar index on a column.

        Scalar indices, like vector indices, can be used to speed up scans.  A scalar
        index can speed up scans that contain filter expressions on the indexed column.
        For example, the following scan will be faster if the column ``my_col`` has
        a scalar index:


            import lancedb

            db = lancedb.connect("/data/lance")
            img_table = db.open_table("images")
            my_df = img_table.search().where("my_col = 7", prefilter=True).to_pandas()

        Scalar indices can also speed up scans containing a vector search and a
        prefilter:

            import lancedb

            db = lancedb.connect("/data/lance")
            img_table = db.open_table("images")
            img_table.search([1, 2, 3, 4], vector_column_name="vector")
                .where("my_col != 7", prefilter=True)
                .to_pandas()

        Scalar indices can only speed up scans for basic filters using
        equality, comparison, range (e.g. ``my_col BETWEEN 0 AND 100``), and set
        membership (e.g. `my_col IN (0, 1, 2)`)

        Scalar indices can be used if the filter contains multiple indexed columns and
        the filter criteria are AND'd or OR'd together
        (e.g. ``my_col < 0 AND other_col> 100``)

        Scalar indices may be used if the filter contains non-indexed columns but,
        depending on the structure of the filter, they may not be usable.  For example,
        if the column ``not_indexed`` does not have a scalar index then the filter
        ``my_col = 0 OR not_indexed = 1`` will not be able to use any scalar index on
        ``my_col``.

        **Experimental API**

        Parameters
        ----------
        column : str
            The column to be indexed.  Must be a boolean, integer, float,
            or string column.
        replace : bool, default True
            Replace the existing index if it exists.
        index_type: Literal["BTREE", "BITMAP", "LABEL_LIST"], default "BTREE"
            The type of index to create.

        Examples
        --------


            import lance

            dataset = lance.dataset("./images.lance")
            dataset.create_scalar_index("category")
        """
        raise NotImplementedError

    def create_fts_index(
        self,
        field_names: Union[str, List[str]],
        ordering_field_names: Union[str, List[str]] = None,
        *,
        replace: bool = False,
        with_position: bool = True,
        writer_heap_size: Optional[int] = 1024 * 1024 * 1024,
        tokenizer_name: str = "default",
        use_tantivy: bool = True,
    ):
        """Create a full-text search index on the table.

        Warning - this API is highly experimental and is highly likely to change
        in the future.

        Parameters
        ----------
        field_names: str or list of str
            The name(s) of the field to index.
            can be only str if use_tantivy=True for now.
        replace: bool, default False
            If True, replace the existing index if it exists. Note that this is
            not yet an atomic operation; the index will be temporarily
            unavailable while the new index is being created.
        writer_heap_size: int, default 1GB
            Only available with use_tantivy=True
        ordering_field_names:
            A list of unsigned type fields to index to optionally order
            results on at search time.
            only available with use_tantivy=True
        tokenizer_name: str, default "default"
            The tokenizer to use for the index. Can be "raw", "default" or the 2 letter
            language code followed by "_stem". So for english it would be "en_stem".
            For available languages see: https://docs.rs/tantivy/latest/tantivy/tokenizer/enum.Language.html
            only available with use_tantivy=True for now
        use_tantivy: bool, default True
            If True, use the legacy full-text search implementation based on tantivy.
            If False, use the new full-text search implementation based on lance-index.
        with_position: bool, default True
            Only available with use_tantivy=False
            If False, do not store the positions of the terms in the text.
            This can reduce the size of the index and improve indexing speed.
            But it will raise an exception for phrase queries.

        """
        raise NotImplementedError

    @abstractmethod
    def add(
        self,
        data: DATA,
        mode: str = "append",
        on_bad_vectors: str = "error",
        fill_value: float = 0.0,
    ):
        """Add more data to the [Table](Table).

        Parameters
        ----------
        data: DATA
            The data to insert into the table. Acceptable types are:

            - dict or list-of-dict

            - pandas.DataFrame

            - pyarrow.Table or pyarrow.RecordBatch
        mode: str
            The mode to use when writing the data. Valid values are
            "append" and "overwrite".
        on_bad_vectors: str, default "error"
            What to do if any of the vectors are not the same size or contains NaNs.
            One of "error", "drop", "fill".
        fill_value: float, default 0.
            The value to use when filling vectors. Only used if on_bad_vectors="fill".

        """
        raise NotImplementedError

    def merge_insert(self, on: Union[str, Iterable[str]]) -> LanceMergeInsertBuilder:
        """
        Returns a [`LanceMergeInsertBuilder`][lancedb.merge.LanceMergeInsertBuilder]
        that can be used to create a "merge insert" operation

        This operation can add rows, update rows, and remove rows all in a single
        transaction. It is a very generic tool that can be used to create
        behaviors like "insert if not exists", "update or insert (i.e. upsert)",
        or even replace a portion of existing data with new data (e.g. replace
        all data where month="january")

        The merge insert operation works by combining new data from a
        **source table** with existing data in a **target table** by using a
        join.  There are three categories of records.

        "Matched" records are records that exist in both the source table and
        the target table. "Not matched" records exist only in the source table
        (e.g. these are new data) "Not matched by source" records exist only
        in the target table (this is old data)

        The builder returned by this method can be used to customize what
        should happen for each category of data.

        Please note that the data may appear to be reordered as part of this
        operation.  This is because updated rows will be deleted from the
        dataset and then reinserted at the end with the new values.

        Parameters
        ----------

        on: Union[str, Iterable[str]]
            A column (or columns) to join on.  This is how records from the
            source table and target table are matched.  Typically this is some
            kind of key or id column.

        Examples
        --------
        >>> import lancedb
        >>> data = pa.table({"a": [2, 1, 3], "b": ["a", "b", "c"]})
        >>> db = lancedb.connect("./.lancedb")
        >>> table = db.create_table("my_table", data)
        >>> new_data = pa.table({"a": [2, 3, 4], "b": ["x", "y", "z"]})
        >>> # Perform a "upsert" operation
        >>> table.merge_insert("a")             \\
        ...      .when_matched_update_all()     \\
        ...      .when_not_matched_insert_all() \\
        ...      .execute(new_data)
        >>> # The order of new rows is non-deterministic since we use
        >>> # a hash-join as part of this operation and so we sort here
        >>> table.to_arrow().sort_by("a").to_pandas()
           a  b
        0  1  b
        1  2  x
        2  3  y
        3  4  z
        """
        on = [on] if isinstance(on, str) else list(on.iter())

        return LanceMergeInsertBuilder(self, on)

    @abstractmethod
    def search(
        self,
        query: Optional[Union[VEC, str, "PIL.Image.Image", Tuple]] = None,
        vector_column_name: Optional[str] = None,
        query_type: QueryType = "auto",
        ordering_field_name: Optional[str] = None,
        fts_columns: Optional[Union[str, List[str]]] = None,
    ) -> LanceQueryBuilder:
        """Create a search query to find the nearest neighbors
        of the given query vector. We currently support [vector search][search]
        and [full-text search][experimental-full-text-search].

        All query options are defined in [Query][lancedb.query.Query].

        Examples
        --------
        >>> import lancedb
        >>> db = lancedb.connect("./.lancedb")
        >>> data = [
        ...    {"original_width": 100, "caption": "bar", "vector": [0.1, 2.3, 4.5]},
        ...    {"original_width": 2000, "caption": "foo",  "vector": [0.5, 3.4, 1.3]},
        ...    {"original_width": 3000, "caption": "test", "vector": [0.3, 6.2, 2.6]}
        ... ]
        >>> table = db.create_table("my_table", data)
        >>> query = [0.4, 1.4, 2.4]
        >>> (table.search(query)
        ...     .where("original_width > 1000", prefilter=True)
        ...     .select(["caption", "original_width", "vector"])
        ...     .limit(2)
        ...     .to_pandas())
          caption  original_width           vector  _distance
        0     foo            2000  [0.5, 3.4, 1.3]   5.220000
        1    test            3000  [0.3, 6.2, 2.6]  23.089996

        Parameters
        ----------
        query: list/np.ndarray/str/PIL.Image.Image, default None
            The targetted vector to search for.

            - *default None*.
            Acceptable types are: list, np.ndarray, PIL.Image.Image

            - If None then the select/where/limit clauses are applied to filter
            the table
        vector_column_name: str, optional
            The name of the vector column to search.

            The vector column needs to be a pyarrow fixed size list type

            - If not specified then the vector column is inferred from
            the table schema

            - If the table has multiple vector columns then the *vector_column_name*
            needs to be specified. Otherwise, an error is raised.
        query_type: str
            *default "auto"*.
            Acceptable types are: "vector", "fts", "hybrid", or "auto"

            - If "auto" then the query type is inferred from the query;

                - If `query` is a list/np.ndarray then the query type is
                "vector";

                - If `query` is a PIL.Image.Image then either do vector search,
                or raise an error if no corresponding embedding function is found.

            - If `query` is a string, then the query type is "vector" if the
            table has embedding functions else the query type is "fts"

        Returns
        -------
        LanceQueryBuilder
            A query builder object representing the query.
            Once executed, the query returns

            - selected columns

            - the vector

            - and also the "_distance" column which is the distance between the query
            vector and the returned vector.
        """
        raise NotImplementedError

    @abstractmethod
    def _execute_query(
        self, query: Query, batch_size: Optional[int] = None
    ) -> pa.RecordBatchReader:
        pass

    @abstractmethod
    def _do_merge(
        self,
        merge: LanceMergeInsertBuilder,
        new_data: DATA,
        on_bad_vectors: str,
        fill_value: float,
    ):
        pass

    @abstractmethod
    def delete(self, where: str):
        """Delete rows from the table.

        This can be used to delete a single row, many rows, all rows, or
        sometimes no rows (if your predicate matches nothing).

        Parameters
        ----------
        where: str
            The SQL where clause to use when deleting rows.

            - For example, 'x = 2' or 'x IN (1, 2, 3)'.

            The filter must not be empty, or it will error.

        Examples
        --------
        >>> import lancedb
        >>> data = [
        ...    {"x": 1, "vector": [1, 2]},
        ...    {"x": 2, "vector": [3, 4]},
        ...    {"x": 3, "vector": [5, 6]}
        ... ]
        >>> db = lancedb.connect("./.lancedb")
        >>> table = db.create_table("my_table", data)
        >>> table.to_pandas()
           x      vector
        0  1  [1.0, 2.0]
        1  2  [3.0, 4.0]
        2  3  [5.0, 6.0]
        >>> table.delete("x = 2")
        >>> table.to_pandas()
           x      vector
        0  1  [1.0, 2.0]
        1  3  [5.0, 6.0]

        If you have a list of values to delete, you can combine them into a
        stringified list and use the `IN` operator:

        >>> to_remove = [1, 5]
        >>> to_remove = ", ".join([str(v) for v in to_remove])
        >>> to_remove
        '1, 5'
        >>> table.delete(f"x IN ({to_remove})")
        >>> table.to_pandas()
           x      vector
        0  3  [5.0, 6.0]
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        where: Optional[str] = None,
        values: Optional[dict] = None,
        *,
        values_sql: Optional[Dict[str, str]] = None,
    ):
        """
        This can be used to update zero to all rows depending on how many
        rows match the where clause. If no where clause is provided, then
        all rows will be updated.

        Either `values` or `values_sql` must be provided. You cannot provide
        both.

        Parameters
        ----------
        where: str, optional
            The SQL where clause to use when updating rows. For example, 'x = 2'
            or 'x IN (1, 2, 3)'. The filter must not be empty, or it will error.
        values: dict, optional
            The values to update. The keys are the column names and the values
            are the values to set.
        values_sql: dict, optional
            The values to update, expressed as SQL expression strings. These can
            reference existing columns. For example, {"x": "x + 1"} will increment
            the x column by 1.

        Examples
        --------
        >>> import lancedb
        >>> import pandas as pd
        >>> data = pd.DataFrame({"x": [1, 2, 3], "vector": [[1, 2], [3, 4], [5, 6]]})
        >>> db = lancedb.connect("./.lancedb")
        >>> table = db.create_table("my_table", data)
        >>> table.to_pandas()
           x      vector
        0  1  [1.0, 2.0]
        1  2  [3.0, 4.0]
        2  3  [5.0, 6.0]
        >>> table.update(where="x = 2", values={"vector": [10, 10]})
        >>> table.to_pandas()
           x        vector
        0  1    [1.0, 2.0]
        1  3    [5.0, 6.0]
        2  2  [10.0, 10.0]
        >>> table.update(values_sql={"x": "x + 1"})
        >>> table.to_pandas()
           x        vector
        0  2    [1.0, 2.0]
        1  4    [5.0, 6.0]
        2  3  [10.0, 10.0]
        """
        raise NotImplementedError

    @abstractmethod
    def cleanup_old_versions(
        self,
        older_than: Optional[timedelta] = None,
        *,
        delete_unverified: bool = False,
    ) -> CleanupStats:
        """
        Clean up old versions of the table, freeing disk space.

        Note: This function is not available in LanceDb Cloud (since LanceDb
        Cloud manages cleanup for you automatically)

        Parameters
        ----------
        older_than: timedelta, default None
            The minimum age of the version to delete. If None, then this defaults
            to two weeks.
        delete_unverified: bool, default False
            Because they may be part of an in-progress transaction, files newer
            than 7 days old are not deleted by default. If you are sure that
            there are no in-progress transactions, then you can set this to True
            to delete all files older than `older_than`.

        Returns
        -------
        CleanupStats
            The stats of the cleanup operation, including how many bytes were
            freed.
        """

    @abstractmethod
    def compact_files(self, *args, **kwargs):
        """
        Run the compaction process on the table.

        Note: This function is not available in LanceDb Cloud (since LanceDb
        Cloud manages compaction for you automatically)

        This can be run after making several small appends to optimize the table
        for faster reads.

        Arguments are passed onto :meth:`lance.dataset.DatasetOptimizer.compact_files`.
        For most cases, the default should be fine.
        """

    @abstractmethod
    def add_columns(self, transforms: Dict[str, str]):
        """
        Add new columns with defined values.

        This is not yet available in LanceDB Cloud.

        Parameters
        ----------
        transforms: Dict[str, str]
            A map of column name to a SQL expression to use to calculate the
            value of the new column. These expressions will be evaluated for
            each row in the table, and can reference existing columns.
        """

    @abstractmethod
    def alter_columns(self, alterations: Iterable[Dict[str, str]]):
        """
        Alter column names and nullability.

        This is not yet available in LanceDB Cloud.

        alterations : Iterable[Dict[str, Any]]
            A sequence of dictionaries, each with the following keys:
            - "path": str
                The column path to alter. For a top-level column, this is the name.
                For a nested column, this is the dot-separated path, e.g. "a.b.c".
            - "name": str, optional
                The new name of the column. If not specified, the column name is
                not changed.
            - "nullable": bool, optional
                Whether the column should be nullable. If not specified, the column
                nullability is not changed. Only non-nullable columns can be changed
                to nullable. Currently, you cannot change a nullable column to
                non-nullable.
        """

    @abstractmethod
    def drop_columns(self, columns: Iterable[str]):
        """
        Drop columns from the table.

        This is not yet available in LanceDB Cloud.

        Parameters
        ----------
        columns : Iterable[str]
            The names of the columns to drop.
        """

    @cached_property
    def _dataset_uri(self) -> str:
        return _table_uri(self._conn.uri, self.name)

    def _get_fts_index_path(self) -> Tuple[str, pa_fs.FileSystem, bool]:
        if get_uri_scheme(self._dataset_uri) != "file":
            return ("", None, False)
        path = join_uri(self._dataset_uri, "_indices", "fts")
        fs, path = fs_from_uri(path)
        index_exists = fs.get_file_info(path).type != pa_fs.FileType.NotFound
        return (path, fs, index_exists)
schema: pa.Schema abstractmethod property
The Arrow Schema of this Table

count_rows(filter: Optional[str] = None) -> int abstractmethod
Count the number of rows in the table.

Parameters:

Name	Type	Description	Default
filter	Optional[str]	A SQL where clause to filter the rows to count.	None
Source code in lancedb/table.py
to_pandas() -> 'pd.DataFrame'
Return the table as a pandas DataFrame.

Returns:

Type	Description
DataFrame	
Source code in lancedb/table.py
to_arrow() -> pa.Table abstractmethod
Return the table as a pyarrow Table.

Returns:

Type	Description
Table	
Source code in lancedb/table.py
create_index(metric='L2', num_partitions=256, num_sub_vectors=96, vector_column_name: str = VECTOR_COLUMN_NAME, replace: bool = True, accelerator: Optional[str] = None, index_cache_size: Optional[int] = None)
Create an index on the table.

Parameters:

Name	Type	Description	Default
metric		The distance metric to use when creating the index. Valid values are "L2", "cosine", or "dot". L2 is euclidean distance.	'L2'
num_partitions		The number of IVF partitions to use when creating the index. Default is 256.	256
num_sub_vectors		The number of PQ sub-vectors to use when creating the index. Default is 96.	96
vector_column_name	str	The vector column name to create the index.	VECTOR_COLUMN_NAME
replace	bool	
If True, replace the existing index if it exists.

If False, raise an error if duplicate index exists.

True
accelerator	Optional[str]	If set, use the given accelerator to create the index. Only support "cuda" for now.	None
index_cache_size	int	The size of the index cache in number of entries. Default value is 256.	None
Source code in lancedb/table.py
create_scalar_index(column: str, *, replace: bool = True, index_type: Literal['BTREE', 'BITMAP', 'LABEL_LIST'] = 'BTREE') abstractmethod
Create a scalar index on a column.

Scalar indices, like vector indices, can be used to speed up scans. A scalar index can speed up scans that contain filter expressions on the indexed column. For example, the following scan will be faster if the column my_col has a scalar index:


import lancedb

db = lancedb.connect("/data/lance")
img_table = db.open_table("images")
my_df = img_table.search().where("my_col = 7", prefilter=True).to_pandas()
Scalar indices can also speed up scans containing a vector search and a prefilter:


import lancedb

db = lancedb.connect("/data/lance")
img_table = db.open_table("images")
img_table.search([1, 2, 3, 4], vector_column_name="vector")
    .where("my_col != 7", prefilter=True)
    .to_pandas()
Scalar indices can only speed up scans for basic filters using equality, comparison, range (e.g. my_col BETWEEN 0 AND 100), and set membership (e.g. my_col IN (0, 1, 2))

Scalar indices can be used if the filter contains multiple indexed columns and the filter criteria are AND'd or OR'd together (e.g. my_col < 0 AND other_col> 100)

Scalar indices may be used if the filter contains non-indexed columns but, depending on the structure of the filter, they may not be usable. For example, if the column not_indexed does not have a scalar index then the filter my_col = 0 OR not_indexed = 1 will not be able to use any scalar index on my_col.

Experimental API

Parameters:

Name	Type	Description	Default
column	str	The column to be indexed. Must be a boolean, integer, float, or string column.	required
replace	bool	Replace the existing index if it exists.	True
index_type	Literal['BTREE', 'BITMAP', 'LABEL_LIST']	The type of index to create.	'BTREE'
Examples:


import lance

dataset = lance.dataset("./images.lance")
dataset.create_scalar_index("category")
Source code in lancedb/table.py
create_fts_index(field_names: Union[str, List[str]], ordering_field_names: Union[str, List[str]] = None, *, replace: bool = False, with_position: bool = True, writer_heap_size: Optional[int] = 1024 * 1024 * 1024, tokenizer_name: str = 'default', use_tantivy: bool = True)
Create a full-text search index on the table.

Warning - this API is highly experimental and is highly likely to change in the future.

Parameters:

Name	Type	Description	Default
field_names	Union[str, List[str]]	The name(s) of the field to index. can be only str if use_tantivy=True for now.	required
replace	bool	If True, replace the existing index if it exists. Note that this is not yet an atomic operation; the index will be temporarily unavailable while the new index is being created.	False
writer_heap_size	Optional[int]	Only available with use_tantivy=True	1024 * 1024 * 1024
ordering_field_names	Union[str, List[str]]	A list of unsigned type fields to index to optionally order results on at search time. only available with use_tantivy=True	None
tokenizer_name	str	The tokenizer to use for the index. Can be "raw", "default" or the 2 letter language code followed by "_stem". So for english it would be "en_stem". For available languages see: https://docs.rs/tantivy/latest/tantivy/tokenizer/enum.Language.html only available with use_tantivy=True for now	'default'
use_tantivy	bool	If True, use the legacy full-text search implementation based on tantivy. If False, use the new full-text search implementation based on lance-index.	True
with_position	bool	Only available with use_tantivy=False If False, do not store the positions of the terms in the text. This can reduce the size of the index and improve indexing speed. But it will raise an exception for phrase queries.	True
Source code in lancedb/table.py
add(data: DATA, mode: str = 'append', on_bad_vectors: str = 'error', fill_value: float = 0.0) abstractmethod
Add more data to the Table.

Parameters:

Name	Type	Description	Default
data	DATA	The data to insert into the table. Acceptable types are:
dict or list-of-dict

pandas.DataFrame

pyarrow.Table or pyarrow.RecordBatch

required
mode	str	The mode to use when writing the data. Valid values are "append" and "overwrite".	'append'
on_bad_vectors	str	What to do if any of the vectors are not the same size or contains NaNs. One of "error", "drop", "fill".	'error'
fill_value	float	The value to use when filling vectors. Only used if on_bad_vectors="fill".	0.0
Source code in lancedb/table.py
merge_insert(on: Union[str, Iterable[str]]) -> LanceMergeInsertBuilder
Returns a LanceMergeInsertBuilder that can be used to create a "merge insert" operation

This operation can add rows, update rows, and remove rows all in a single transaction. It is a very generic tool that can be used to create behaviors like "insert if not exists", "update or insert (i.e. upsert)", or even replace a portion of existing data with new data (e.g. replace all data where month="january")

The merge insert operation works by combining new data from a source table with existing data in a target table by using a join. There are three categories of records.

"Matched" records are records that exist in both the source table and the target table. "Not matched" records exist only in the source table (e.g. these are new data) "Not matched by source" records exist only in the target table (this is old data)

The builder returned by this method can be used to customize what should happen for each category of data.

Please note that the data may appear to be reordered as part of this operation. This is because updated rows will be deleted from the dataset and then reinserted at the end with the new values.

Parameters:

Name	Type	Description	Default
on	Union[str, Iterable[str]]	A column (or columns) to join on. This is how records from the source table and target table are matched. Typically this is some kind of key or id column.	required
Examples:


>>> import lancedb
>>> data = pa.table({"a": [2, 1, 3], "b": ["a", "b", "c"]})
>>> db = lancedb.connect("./.lancedb")
>>> table = db.create_table("my_table", data)
>>> new_data = pa.table({"a": [2, 3, 4], "b": ["x", "y", "z"]})
>>> # Perform a "upsert" operation
>>> table.merge_insert("a")             \
...      .when_matched_update_all()     \
...      .when_not_matched_insert_all() \
...      .execute(new_data)
>>> # The order of new rows is non-deterministic since we use
>>> # a hash-join as part of this operation and so we sort here
>>> table.to_arrow().sort_by("a").to_pandas()
   a  b
0  1  b
1  2  x
2  3  y
3  4  z
Source code in lancedb/table.py
search(query: Optional[Union[VEC, str, 'PIL.Image.Image', Tuple]] = None, vector_column_name: Optional[str] = None, query_type: QueryType = 'auto', ordering_field_name: Optional[str] = None, fts_columns: Optional[Union[str, List[str]]] = None) -> LanceQueryBuilder abstractmethod
Create a search query to find the nearest neighbors of the given query vector. We currently support vector search and [full-text search][experimental-full-text-search].

All query options are defined in Query.

Examples:


>>> import lancedb
>>> db = lancedb.connect("./.lancedb")
>>> data = [
...    {"original_width": 100, "caption": "bar", "vector": [0.1, 2.3, 4.5]},
...    {"original_width": 2000, "caption": "foo",  "vector": [0.5, 3.4, 1.3]},
...    {"original_width": 3000, "caption": "test", "vector": [0.3, 6.2, 2.6]}
... ]
>>> table = db.create_table("my_table", data)
>>> query = [0.4, 1.4, 2.4]
>>> (table.search(query)
...     .where("original_width > 1000", prefilter=True)
...     .select(["caption", "original_width", "vector"])
...     .limit(2)
...     .to_pandas())
  caption  original_width           vector  _distance
0     foo            2000  [0.5, 3.4, 1.3]   5.220000
1    test            3000  [0.3, 6.2, 2.6]  23.089996
Parameters:

Name	Type	Description	Default
query	Optional[Union[VEC, str, 'PIL.Image.Image', Tuple]]	The targetted vector to search for.
default None. Acceptable types are: list, np.ndarray, PIL.Image.Image

If None then the select/where/limit clauses are applied to filter the table

None
vector_column_name	Optional[str]	The name of the vector column to search.
The vector column needs to be a pyarrow fixed size list type

If not specified then the vector column is inferred from the table schema

If the table has multiple vector columns then the vector_column_name needs to be specified. Otherwise, an error is raised.

None
query_type	QueryType	default "auto". Acceptable types are: "vector", "fts", "hybrid", or "auto"
If "auto" then the query type is inferred from the query;

If query is a list/np.ndarray then the query type is "vector";

If query is a PIL.Image.Image then either do vector search, or raise an error if no corresponding embedding function is found.

If query is a string, then the query type is "vector" if the table has embedding functions else the query type is "fts"

'auto'
Returns:

Type	Description
LanceQueryBuilder	A query builder object representing the query. Once executed, the query returns
selected columns

the vector

and also the "_distance" column which is the distance between the query vector and the returned vector.

Source code in lancedb/table.py
delete(where: str) abstractmethod
Delete rows from the table.

This can be used to delete a single row, many rows, all rows, or sometimes no rows (if your predicate matches nothing).

Parameters:

Name	Type	Description	Default
where	str	The SQL where clause to use when deleting rows.
For example, 'x = 2' or 'x IN (1, 2, 3)'.
The filter must not be empty, or it will error.

required
Examples:


>>> import lancedb
>>> data = [
...    {"x": 1, "vector": [1, 2]},
...    {"x": 2, "vector": [3, 4]},
...    {"x": 3, "vector": [5, 6]}
... ]
>>> db = lancedb.connect("./.lancedb")
>>> table = db.create_table("my_table", data)
>>> table.to_pandas()
   x      vector
0  1  [1.0, 2.0]
1  2  [3.0, 4.0]
2  3  [5.0, 6.0]
>>> table.delete("x = 2")
>>> table.to_pandas()
   x      vector
0  1  [1.0, 2.0]
1  3  [5.0, 6.0]
If you have a list of values to delete, you can combine them into a stringified list and use the IN operator:


>>> to_remove = [1, 5]
>>> to_remove = ", ".join([str(v) for v in to_remove])
>>> to_remove
'1, 5'
>>> table.delete(f"x IN ({to_remove})")
>>> table.to_pandas()
   x      vector
0  3  [5.0, 6.0]
Source code in lancedb/table.py
update(where: Optional[str] = None, values: Optional[dict] = None, *, values_sql: Optional[Dict[str, str]] = None) abstractmethod
This can be used to update zero to all rows depending on how many rows match the where clause. If no where clause is provided, then all rows will be updated.

Either values or values_sql must be provided. You cannot provide both.

Parameters:

Name	Type	Description	Default
where	Optional[str]	The SQL where clause to use when updating rows. For example, 'x = 2' or 'x IN (1, 2, 3)'. The filter must not be empty, or it will error.	None
values	Optional[dict]	The values to update. The keys are the column names and the values are the values to set.	None
values_sql	Optional[Dict[str, str]]	The values to update, expressed as SQL expression strings. These can reference existing columns. For example, {"x": "x + 1"} will increment the x column by 1.	None
Examples:


>>> import lancedb
>>> import pandas as pd
>>> data = pd.DataFrame({"x": [1, 2, 3], "vector": [[1, 2], [3, 4], [5, 6]]})
>>> db = lancedb.connect("./.lancedb")
>>> table = db.create_table("my_table", data)
>>> table.to_pandas()
   x      vector
0  1  [1.0, 2.0]
1  2  [3.0, 4.0]
2  3  [5.0, 6.0]
>>> table.update(where="x = 2", values={"vector": [10, 10]})
>>> table.to_pandas()
   x        vector
0  1    [1.0, 2.0]
1  3    [5.0, 6.0]
2  2  [10.0, 10.0]
>>> table.update(values_sql={"x": "x + 1"})
>>> table.to_pandas()
   x        vector
0  2    [1.0, 2.0]
1  4    [5.0, 6.0]
2  3  [10.0, 10.0]
Source code in lancedb/table.py
cleanup_old_versions(older_than: Optional[timedelta] = None, *, delete_unverified: bool = False) -> CleanupStats abstractmethod
Clean up old versions of the table, freeing disk space.

Note: This function is not available in LanceDb Cloud (since LanceDb Cloud manages cleanup for you automatically)

Parameters:

Name	Type	Description	Default
older_than	Optional[timedelta]	The minimum age of the version to delete. If None, then this defaults to two weeks.	None
delete_unverified	bool	Because they may be part of an in-progress transaction, files newer than 7 days old are not deleted by default. If you are sure that there are no in-progress transactions, then you can set this to True to delete all files older than older_than.	False
Returns:

Type	Description
CleanupStats	The stats of the cleanup operation, including how many bytes were freed.
Source code in lancedb/table.py
compact_files(*args, **kwargs) abstractmethod
Run the compaction process on the table.

Note: This function is not available in LanceDb Cloud (since LanceDb Cloud manages compaction for you automatically)

This can be run after making several small appends to optimize the table for faster reads.

Arguments are passed onto :meth:lance.dataset.DatasetOptimizer.compact_files. For most cases, the default should be fine.

Source code in lancedb/table.py
add_columns(transforms: Dict[str, str]) abstractmethod
Add new columns with defined values.

This is not yet available in LanceDB Cloud.

Parameters:

Name	Type	Description	Default
transforms	Dict[str, str]	A map of column name to a SQL expression to use to calculate the value of the new column. These expressions will be evaluated for each row in the table, and can reference existing columns.	required
Source code in lancedb/table.py
alter_columns(alterations: Iterable[Dict[str, str]]) abstractmethod
Alter column names and nullability.

This is not yet available in LanceDB Cloud.

alterations : Iterable[Dict[str, Any]] A sequence of dictionaries, each with the following keys: - "path": str The column path to alter. For a top-level column, this is the name. For a nested column, this is the dot-separated path, e.g. "a.b.c". - "name": str, optional The new name of the column. If not specified, the column name is not changed. - "nullable": bool, optional Whether the column should be nullable. If not specified, the column nullability is not changed. Only non-nullable columns can be changed to nullable. Currently, you cannot change a nullable column to non-nullable.

Source code in lancedb/table.py
drop_columns(columns: Iterable[str]) abstractmethod
Drop columns from the table.

This is not yet available in LanceDB Cloud.

Parameters:

Name	Type	Description	Default
columns	Iterable[str]	The names of the columns to drop.	required
Source code in lancedb/table.py
Querying (Synchronous)
lancedb.query.Query
Bases: BaseModel

The LanceDB Query

Attributes:

Name	Type	Description
vector	List[float]	the vector to search for
filter	Optional[str]	sql filter to refine the query with, optional
prefilter	bool	if True then apply the filter before vector search
k	int	top k results to return
metric	str	the distance metric between a pair of vectors,
can support L2 (default), Cosine and Dot. metric definitions

columns	Optional[List[str]]	which columns to return in the results
nprobes	int	The number of probes used - optional
A higher number makes search more accurate but also slower.

See discussion in Querying an ANN Index for tuning advice.

refine_factor	Optional[int]	Refine the results by reading extra elements and re-ranking them in memory.
A higher number makes search more accurate but also slower.

See discussion in Querying an ANN Index for tuning advice.

offset	int	The offset to start fetching results from
Source code in lancedb/query.py
lancedb.query.LanceQueryBuilder
Bases: ABC

An abstract query builder. Subclasses are defined for vector search, full text search, hybrid, and plain SQL filtering.

Source code in lancedb/query.py
create(table: 'Table', query: Optional[Union[np.ndarray, str, 'PIL.Image.Image', Tuple]], query_type: str, vector_column_name: str, ordering_field_name: Optional[str] = None, fts_columns: Union[str, List[str]] = []) -> LanceQueryBuilder classmethod
Create a query builder based on the given query and query type.

Parameters:

Name	Type	Description	Default
table	'Table'	The table to query.	required
query	Optional[Union[ndarray, str, 'PIL.Image.Image', Tuple]]	The query to use. If None, an empty query builder is returned which performs simple SQL filtering.	required
query_type	str	The type of query to perform. One of "vector", "fts", "hybrid", or "auto". If "auto", the query type is inferred based on the query.	required
vector_column_name	str	The name of the vector column to use for vector search.	required
Source code in lancedb/query.py
to_df() -> 'pd.DataFrame'
Deprecated alias for to_pandas(). Please use to_pandas() instead.

Execute the query and return the results as a pandas DataFrame. In addition to the selected columns, LanceDB also returns a vector and also the "_distance" column which is the distance between the query vector and the returned vector.

Source code in lancedb/query.py
to_pandas(flatten: Optional[Union[int, bool]] = None) -> 'pd.DataFrame'
Execute the query and return the results as a pandas DataFrame. In addition to the selected columns, LanceDB also returns a vector and also the "_distance" column which is the distance between the query vector and the returned vector.

Parameters:

Name	Type	Description	Default
flatten	Optional[Union[int, bool]]	If flatten is True, flatten all nested columns. If flatten is an integer, flatten the nested columns up to the specified depth. If unspecified, do not flatten the nested columns.	None
Source code in lancedb/query.py
to_arrow() -> pa.Table abstractmethod
Execute the query and return the results as an Apache Arrow Table.

In addition to the selected columns, LanceDB also returns a vector and also the "_distance" column which is the distance between the query vector and the returned vectors.

Source code in lancedb/query.py
to_list() -> List[dict]
Execute the query and return the results as a list of dictionaries.

Each list entry is a dictionary with the selected column names as keys, or all table columns if select is not called. The vector and the "_distance" fields are returned whether or not they're explicitly selected.

Source code in lancedb/query.py
to_pydantic(model: Type[LanceModel]) -> List[LanceModel]
Return the table as a list of pydantic models.

Parameters:

Name	Type	Description	Default
model	Type[LanceModel]	The pydantic model to use.	required
Returns:

Type	Description
List[LanceModel]	
Source code in lancedb/query.py
to_polars() -> 'pl.DataFrame'
Execute the query and return the results as a Polars DataFrame. In addition to the selected columns, LanceDB also returns a vector and also the "_distance" column which is the distance between the query vector and the returned vector.

Source code in lancedb/query.py
limit(limit: Union[int, None]) -> LanceQueryBuilder
Set the maximum number of results to return.

Parameters:

Name	Type	Description	Default
limit	Union[int, None]	The maximum number of results to return. By default the query is limited to the first 10. Call this method and pass 0, a negative value, or None to remove the limit. WARNING if you have a large dataset, removing the limit can potentially result in reading a large amount of data into memory and cause out of memory issues.	required
Returns:

Type	Description
LanceQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
offset(offset: int) -> LanceQueryBuilder
Set the offset for the results.

Parameters:

Name	Type	Description	Default
offset	int	The offset to start fetching results from.	required
Returns:

Type	Description
LanceQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
select(columns: Union[list[str], dict[str, str]]) -> LanceQueryBuilder
Set the columns to return.

Parameters:

Name	Type	Description	Default
columns	Union[list[str], dict[str, str]]	List of column names to be fetched. Or a dictionary of column names to SQL expressions. All columns are fetched if None or unspecified.	required
Returns:

Type	Description
LanceQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
where(where: str, prefilter: bool = False) -> LanceQueryBuilder
Set the where clause.

Parameters:

Name	Type	Description	Default
where	str	The where clause which is a valid SQL where clause. See Lance filter pushdown <https://lancedb.github.io/lance/read_and_write.html#filter-push-down>_ for valid SQL expressions.	required
prefilter	bool	If True, apply the filter before vector search, otherwise the filter is applied on the result of vector search. This feature is EXPERIMENTAL and may be removed and modified without warning in the future.	False
Returns:

Type	Description
LanceQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
with_row_id(with_row_id: bool) -> LanceQueryBuilder
Set whether to return row ids.

Parameters:

Name	Type	Description	Default
with_row_id	bool	If True, return _rowid column in the results.	required
Returns:

Type	Description
LanceQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
explain_plan(verbose: Optional[bool] = False) -> str
Return the execution plan for this query.

Examples:


>>> import lancedb
>>> db = lancedb.connect("./.lancedb")
>>> table = db.create_table("my_table", [{"vector": [99, 99]}])
>>> query = [100, 100]
>>> plan = table.search(query).explain_plan(True)
>>> print(plan)
ProjectionExec: expr=[vector@0 as vector, _distance@2 as _distance]
  FilterExec: _distance@2 IS NOT NULL
    SortExec: TopK(fetch=10), expr=[_distance@2 ASC NULLS LAST], preserve_partitioning=[false]
      KNNVectorDistance: metric=l2
        LanceScan: uri=..., projection=[vector], row_id=true, row_addr=false, ordered=false
Parameters:

Name	Type	Description	Default
verbose	bool	Use a verbose output format.	False
Returns:

Name	Type	Description
plan	str	
Source code in lancedb/query.py
vector(vector: Union[np.ndarray, list]) -> LanceQueryBuilder
Set the vector to search for.

Parameters:

Name	Type	Description	Default
vector	Union[ndarray, list]	The vector to search for.	required
Returns:

Type	Description
LanceQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
text(text: str) -> LanceQueryBuilder
Set the text to search for.

Parameters:

Name	Type	Description	Default
text	str	The text to search for.	required
Returns:

Type	Description
LanceQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
rerank(reranker: Reranker) -> LanceQueryBuilder abstractmethod
Rerank the results using the specified reranker.

Parameters:

Name	Type	Description	Default
reranker	Reranker	The reranker to use.	required
Returns:

Type	Description
The LanceQueryBuilder object.	
Source code in lancedb/query.py
lancedb.query.LanceVectorQueryBuilder
Bases: LanceQueryBuilder

Examples:


>>> import lancedb
>>> data = [{"vector": [1.1, 1.2], "b": 2},
...         {"vector": [0.5, 1.3], "b": 4},
...         {"vector": [0.4, 0.4], "b": 6},
...         {"vector": [0.4, 0.4], "b": 10}]
>>> db = lancedb.connect("./.lancedb")
>>> table = db.create_table("my_table", data=data)
>>> (table.search([0.4, 0.4])
...       .metric("cosine")
...       .where("b < 10")
...       .select(["b", "vector"])
...       .limit(2)
...       .to_pandas())
   b      vector  _distance
0  6  [0.4, 0.4]        0.0
Source code in lancedb/query.py
metric(metric: Literal['L2', 'cosine']) -> LanceVectorQueryBuilder
Set the distance metric to use.

Parameters:

Name	Type	Description	Default
metric	Literal['L2', 'cosine']	The distance metric to use. By default "L2" is used.	required
Returns:

Type	Description
LanceVectorQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
nprobes(nprobes: int) -> LanceVectorQueryBuilder
Set the number of probes to use.

Higher values will yield better recall (more likely to find vectors if they exist) at the expense of latency.

See discussion in Querying an ANN Index for tuning advice.

Parameters:

Name	Type	Description	Default
nprobes	int	The number of probes to use.	required
Returns:

Type	Description
LanceVectorQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
refine_factor(refine_factor: int) -> LanceVectorQueryBuilder
Set the refine factor to use, increasing the number of vectors sampled.

As an example, a refine factor of 2 will sample 2x as many vectors as requested, re-ranks them, and returns the top half most relevant results.

See discussion in Querying an ANN Index for tuning advice.

Parameters:

Name	Type	Description	Default
refine_factor	int	The refine factor to use.	required
Returns:

Type	Description
LanceVectorQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
to_arrow() -> pa.Table
Execute the query and return the results as an Apache Arrow Table.

In addition to the selected columns, LanceDB also returns a vector and also the "_distance" column which is the distance between the query vector and the returned vectors.

Source code in lancedb/query.py
to_batches(batch_size: Optional[int] = None) -> pa.RecordBatchReader
Execute the query and return the result as a RecordBatchReader object.

Parameters:

Name	Type	Description	Default
batch_size	Optional[int]	The maximum number of selected records in a RecordBatch object.	None
Returns:

Type	Description
RecordBatchReader	
Source code in lancedb/query.py
where(where: str, prefilter: bool = False) -> LanceVectorQueryBuilder
Set the where clause.

Parameters:

Name	Type	Description	Default
where	str	The where clause which is a valid SQL where clause. See Lance filter pushdown <https://lancedb.github.io/lance/read_and_write.html#filter-push-down>_ for valid SQL expressions.	required
prefilter	bool	If True, apply the filter before vector search, otherwise the filter is applied on the result of vector search. This feature is EXPERIMENTAL and may be removed and modified without warning in the future.	False
Returns:

Type	Description
LanceQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
rerank(reranker: Reranker, query_string: Optional[str] = None) -> LanceVectorQueryBuilder
Rerank the results using the specified reranker.

Parameters:

Name	Type	Description	Default
reranker	Reranker	The reranker to use.	required
query_string	Optional[str]	The query to use for reranking. This needs to be specified explicitly here as the query used for vector search may already be vectorized and the reranker requires a string query. This is only required if the query used for vector search is not a string. Note: This doesn't yet support the case where the query is multimodal or a list of vectors.	None
Returns:

Type	Description
LanceVectorQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
lancedb.query.LanceFtsQueryBuilder
Bases: LanceQueryBuilder

A builder for full text search for LanceDB.

Source code in lancedb/query.py
phrase_query(phrase_query: bool = True) -> LanceFtsQueryBuilder
Set whether to use phrase query.

Parameters:

Name	Type	Description	Default
phrase_query	bool	If True, then the query will be wrapped in quotes and double quotes replaced by single quotes.	True
Returns:

Type	Description
LanceFtsQueryBuilder	The LanceFtsQueryBuilder object.
Source code in lancedb/query.py
rerank(reranker: Reranker) -> LanceFtsQueryBuilder
Rerank the results using the specified reranker.

Parameters:

Name	Type	Description	Default
reranker	Reranker	The reranker to use.	required
Returns:

Type	Description
LanceFtsQueryBuilder	The LanceQueryBuilder object.
Source code in lancedb/query.py
lancedb.query.LanceHybridQueryBuilder
Bases: LanceQueryBuilder

A query builder that performs hybrid vector and full text search. Results are combined and reranked based on the specified reranker. By default, the results are reranked using the RRFReranker, which uses reciprocal rank fusion score for reranking.

To make the vector and fts results comparable, the scores are normalized. Instead of normalizing scores, the normalize parameter can be set to "rank" in the rerank method to convert the scores to ranks and then normalize them.

Source code in lancedb/query.py
phrase_query(phrase_query: bool = True) -> LanceHybridQueryBuilder
Set whether to use phrase query.

Parameters:

Name	Type	Description	Default
phrase_query	bool	If True, then the query will be wrapped in quotes and double quotes replaced by single quotes.	True
Returns:

Type	Description
LanceHybridQueryBuilder	The LanceHybridQueryBuilder object.
Source code in lancedb/query.py
rerank(normalize='score', reranker: Reranker = RRFReranker()) -> LanceHybridQueryBuilder
Rerank the hybrid search results using the specified reranker. The reranker must be an instance of Reranker class.

Parameters:

Name	Type	Description	Default
normalize		The method to normalize the scores. Can be "rank" or "score". If "rank", the scores are converted to ranks and then normalized. If "score", the scores are normalized directly.	'score'
reranker	Reranker	The reranker to use. Must be an instance of Reranker class.	RRFReranker()
Returns:

Type	Description
LanceHybridQueryBuilder	The LanceHybridQueryBuilder object.
Source code in lancedb/query.py
nprobes(nprobes: int) -> LanceHybridQueryBuilder
Set the number of probes to use for vector search.

Higher values will yield better recall (more likely to find vectors if they exist) at the expense of latency.

Parameters:

Name	Type	Description	Default
nprobes	int	The number of probes to use.	required
Returns:

Type	Description
LanceHybridQueryBuilder	The LanceHybridQueryBuilder object.
Source code in lancedb/query.py
refine_factor(refine_factor: int) -> LanceHybridQueryBuilder
Refine the vector search results by reading extra elements and re-ranking them in memory.

Parameters:

Name	Type	Description	Default
refine_factor	int	The refine factor to use.	required
Returns:

Type	Description
LanceHybridQueryBuilder	The LanceHybridQueryBuilder object.
Source code in lancedb/query.py
Embeddings
lancedb.embeddings.registry.EmbeddingFunctionRegistry
This is a singleton class used to register embedding functions and fetch them by name. It also handles serializing and deserializing. You can implement your own embedding function by subclassing EmbeddingFunction or TextEmbeddingFunction and registering it with the registry.

NOTE: Here TEXT is a type alias for Union[str, List[str], pa.Array, pa.ChunkedArray, np.ndarray]

Examples:


>>> registry = EmbeddingFunctionRegistry.get_instance()
>>> @registry.register("my-embedding-function")
... class MyEmbeddingFunction(EmbeddingFunction):
...     def ndims(self) -> int:
...         return 128
...
...     def compute_query_embeddings(self, query: str, *args, **kwargs):
...         return self.compute_source_embeddings(query, *args, **kwargs)
...
...     def compute_source_embeddings(self, texts, *args, **kwargs):
...         return [np.random.rand(self.ndims()) for _ in range(len(texts))]
...
>>> registry.get("my-embedding-function")
<class 'lancedb.embeddings.registry.MyEmbeddingFunction'>
Source code in lancedb/embeddings/registry.py
register(alias: str = None)
This creates a decorator that can be used to register an EmbeddingFunction.

Parameters:

Name	Type	Description	Default
alias	Optional[str]	a human friendly name for the embedding function. If not provided, the class name will be used.	None
Source code in lancedb/embeddings/registry.py
reset()
Reset the registry to its initial state

Source code in lancedb/embeddings/registry.py
get(name: str)
Fetch an embedding function class by name

Parameters:

Name	Type	Description	Default
name	str	The name of the embedding function to fetch Either the alias or the class name if no alias was provided during registration	required
Source code in lancedb/embeddings/registry.py
parse_functions(metadata: Optional[Dict[bytes, bytes]]) -> Dict[str, EmbeddingFunctionConfig]
Parse the metadata from an arrow table and return a mapping of the vector column to the embedding function and source column

Parameters:

Name	Type	Description	Default
metadata	Optional[Dict[bytes, bytes]]	The metadata from an arrow table. Note that the keys and values are bytes (pyarrow api)	required
Returns:

Name	Type	Description
functions	dict	A mapping of vector column name to embedding function. An empty dict is returned if input is None or does not contain b"embedding_functions".
Source code in lancedb/embeddings/registry.py
function_to_metadata(conf: EmbeddingFunctionConfig)
Convert the given embedding function and source / vector column configs into a config dictionary that can be serialized into arrow metadata

Source code in lancedb/embeddings/registry.py
get_table_metadata(func_list)
Convert a list of embedding functions and source / vector configs into a config dictionary that can be serialized into arrow metadata

Source code in lancedb/embeddings/registry.py
lancedb.embeddings.base.EmbeddingFunction
Bases: BaseModel, ABC

An ABC for embedding functions.

All concrete embedding functions must implement the following: 1. compute_query_embeddings() which takes a query and returns a list of embeddings 2. get_source_embeddings() which returns a list of embeddings for the source column For text data, the two will be the same. For multi-modal data, the source column might be images and the vector column might be text. 3. ndims method which returns the number of dimensions of the vector column

Source code in lancedb/embeddings/base.py
create(**kwargs) classmethod
Create an instance of the embedding function

Source code in lancedb/embeddings/base.py
compute_query_embeddings(*args, **kwargs) -> List[np.array] abstractmethod
Compute the embeddings for a given user query

Source code in lancedb/embeddings/base.py
compute_source_embeddings(*args, **kwargs) -> List[np.array] abstractmethod
Compute the embeddings for the source column in the database

Source code in lancedb/embeddings/base.py
compute_query_embeddings_with_retry(*args, **kwargs) -> List[np.array]
Compute the embeddings for a given user query with retries

Source code in lancedb/embeddings/base.py
compute_source_embeddings_with_retry(*args, **kwargs) -> List[np.array]
Compute the embeddings for the source column in the database with retries

Source code in lancedb/embeddings/base.py
sanitize_input(texts: TEXT) -> Union[List[str], np.ndarray]
Sanitize the input to the embedding function.

Source code in lancedb/embeddings/base.py
ndims() abstractmethod
Return the dimensions of the vector column

Source code in lancedb/embeddings/base.py
SourceField(**kwargs)
Creates a pydantic Field that can automatically annotate the source column for this embedding function

Source code in lancedb/embeddings/base.py
VectorField(**kwargs)
Creates a pydantic Field that can automatically annotate the target vector column for this embedding function

Source code in lancedb/embeddings/base.py
lancedb.embeddings.base.TextEmbeddingFunction
Bases: EmbeddingFunction

A callable ABC for embedding functions that take text as input

Source code in lancedb/embeddings/base.py
generate_embeddings(texts: Union[List[str], np.ndarray], *args, **kwargs) -> List[np.array] abstractmethod
Generate the embeddings for the given texts

Source code in lancedb/embeddings/base.py
lancedb.embeddings.sentence_transformers.SentenceTransformerEmbeddings
Bases: TextEmbeddingFunction

An embedding function that uses the sentence-transformers library

https://huggingface.co/sentence-transformers

Parameters:

Name	Type	Description	Default
name		The name of the model to use.	required
device		The device to use for the model	required
normalize		Whether to normalize the embeddings	required
trust_remote_code		Whether to trust the remote code	required
Source code in lancedb/embeddings/sentence_transformers.py
embedding_model property
Get the sentence-transformers embedding model specified by the name, device, and trust_remote_code. This is cached so that the model is only loaded once per process.

generate_embeddings(texts: Union[List[str], np.ndarray]) -> List[np.array]
Get the embeddings for the given texts

Parameters:

Name	Type	Description	Default
texts	Union[List[str], ndarray]	The texts to embed	required
Source code in lancedb/embeddings/sentence_transformers.py
get_embedding_model()
Get the sentence-transformers embedding model specified by the name, device, and trust_remote_code. This is cached so that the model is only loaded once per process.

TODO: use lru_cache instead with a reasonable/configurable maxsize

Source code in lancedb/embeddings/sentence_transformers.py
lancedb.embeddings.openai.OpenAIEmbeddings
Bases: TextEmbeddingFunction

An embedding function that uses the OpenAI API

https://platform.openai.com/docs/guides/embeddings

This can also be used for open source models that are compatible with the OpenAI API.

Notes
If you're running an Ollama server locally, you can just override the base_url parameter and provide the Ollama embedding model you want to use (https://ollama.com/library):


from lancedb.embeddings import get_registry
openai = get_registry().get("openai")
embedding_function = openai.create(
    name="<ollama-embedding-model-name>",
    base_url="http://localhost:11434",
    )
Source code in lancedb/embeddings/openai.py
generate_embeddings(texts: Union[List[str], np.ndarray]) -> List[np.array]
Get the embeddings for the given texts

Parameters:

Name	Type	Description	Default
texts	Union[List[str], ndarray]	The texts to embed	required
Source code in lancedb/embeddings/openai.py
lancedb.embeddings.open_clip.OpenClipEmbeddings
Bases: EmbeddingFunction

An embedding function that uses the OpenClip API For multi-modal text-to-image search

https://github.com/mlfoundations/open_clip

Source code in lancedb/embeddings/open_clip.py
compute_query_embeddings(query: Union[str, PIL.Image.Image], *args, **kwargs) -> List[np.ndarray]
Compute the embeddings for a given user query

Parameters:

Name	Type	Description	Default
query	Union[str, Image]	The query to embed. A query can be either text or an image.	required
Source code in lancedb/embeddings/open_clip.py
sanitize_input(images: IMAGES) -> Union[List[bytes], np.ndarray]
Sanitize the input to the embedding function.

Source code in lancedb/embeddings/open_clip.py
compute_source_embeddings(images: IMAGES, *args, **kwargs) -> List[np.array]
Get the embeddings for the given images

Source code in lancedb/embeddings/open_clip.py
generate_image_embedding(image: Union[str, bytes, PIL.Image.Image]) -> np.ndarray
Generate the embedding for a single image

Parameters:

Name	Type	Description	Default
image	Union[str, bytes, Image]	The image to embed. If the image is a str, it is treated as a uri. If the image is bytes, it is treated as the raw image bytes.	required
Source code in lancedb/embeddings/open_clip.py
lancedb.embeddings.utils.with_embeddings(func: Callable, data: DATA, column: str = 'text', wrap_api: bool = True, show_progress: bool = False, batch_size: int = 1000) -> pa.Table
Add a vector column to a table using the given embedding function.

The new columns will be called "vector".

Parameters:

Name	Type	Description	Default
func	Callable	A function that takes a list of strings and returns a list of vectors.	required
data	Table or DataFrame	The data to add an embedding column to.	required
column	str	The name of the column to use as input to the embedding function.	"text"
wrap_api	bool	Whether to wrap the embedding function in a retry and rate limiter.	True
show_progress	bool	Whether to show a progress bar.	False
batch_size	int	The number of row values to pass to each call of the embedding function.	1000
Returns:

Type	Description
Table	The input table with a new column called "vector" containing the embeddings.
Source code in lancedb/embeddings/utils.py
Context
lancedb.context.contextualize(raw_df: 'pd.DataFrame') -> Contextualizer
Create a Contextualizer object for the given DataFrame.

Used to create context windows. Context windows are rolling subsets of text data.

The input text column should already be separated into rows that will be the unit of the window. So to create a context window over tokens, start with a DataFrame with one token per row. To create a context window over sentences, start with a DataFrame with one sentence per row.

Examples:


>>> from lancedb.context import contextualize
>>> import pandas as pd
>>> data = pd.DataFrame({
...    'token': ['The', 'quick', 'brown', 'fox', 'jumped', 'over',
...              'the', 'lazy', 'dog', 'I', 'love', 'sandwiches'],
...    'document_id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
... })
window determines how many rows to include in each window. In our case this how many tokens, but depending on the input data, it could be sentences, paragraphs, messages, etc.


>>> contextualize(data).window(3).stride(1).text_col('token').to_pandas()
                token  document_id
0     The quick brown            1
1     quick brown fox            1
2    brown fox jumped            1
3     fox jumped over            1
4     jumped over the            1
5       over the lazy            1
6        the lazy dog            1
7          lazy dog I            1
8          dog I love            1
9   I love sandwiches            2
10    love sandwiches            2
>>> (contextualize(data).window(7).stride(1).min_window_size(7)
...   .text_col('token').to_pandas())
                                  token  document_id
0   The quick brown fox jumped over the            1
1  quick brown fox jumped over the lazy            1
2    brown fox jumped over the lazy dog            1
3        fox jumped over the lazy dog I            1
4       jumped over the lazy dog I love            1
5   over the lazy dog I love sandwiches            1
stride determines how many rows to skip between each window start. This can be used to reduce the total number of windows generated.


>>> contextualize(data).window(4).stride(2).text_col('token').to_pandas()
                    token  document_id
0     The quick brown fox            1
2   brown fox jumped over            1
4    jumped over the lazy            1
6          the lazy dog I            1
8   dog I love sandwiches            1
10        love sandwiches            2
groupby determines how to group the rows. For example, we would like to have context windows that don't cross document boundaries. In this case, we can pass document_id as the group by.


>>> (contextualize(data)
...     .window(4).stride(2).text_col('token').groupby('document_id')
...     .to_pandas())
                   token  document_id
0    The quick brown fox            1
2  brown fox jumped over            1
4   jumped over the lazy            1
6           the lazy dog            1
9      I love sandwiches            2
min_window_size determines the minimum size of the context windows that are generated.This can be used to trim the last few context windows which have size less than min_window_size. By default context windows of size 1 are skipped.


>>> (contextualize(data)
...     .window(6).stride(3).text_col('token').groupby('document_id')
...     .to_pandas())
                             token  document_id
0  The quick brown fox jumped over            1
3     fox jumped over the lazy dog            1
6                     the lazy dog            1
9                I love sandwiches            2

>>> (contextualize(data)
...     .window(6).stride(3).min_window_size(4).text_col('token')
...     .groupby('document_id')
...     .to_pandas())
                             token  document_id
0  The quick brown fox jumped over            1
3     fox jumped over the lazy dog            1
Source code in lancedb/context.py
lancedb.context.Contextualizer
Create context windows from a DataFrame. See lancedb.context.contextualize.

Source code in lancedb/context.py
window(window: int) -> Contextualizer
Set the window size. i.e., how many rows to include in each window.

Parameters:

Name	Type	Description	Default
window	int	The window size.	required
Source code in lancedb/context.py
stride(stride: int) -> Contextualizer
Set the stride. i.e., how many rows to skip between each window.

Parameters:

Name	Type	Description	Default
stride	int	The stride.	required
Source code in lancedb/context.py
groupby(groupby: str) -> Contextualizer
Set the groupby column. i.e., how to group the rows. Windows don't cross groups

Parameters:

Name	Type	Description	Default
groupby	str	The groupby column.	required
Source code in lancedb/context.py
text_col(text_col: str) -> Contextualizer
Set the text column used to make the context window.

Parameters:

Name	Type	Description	Default
text_col	str	The text column.	required
Source code in lancedb/context.py
min_window_size(min_window_size: int) -> Contextualizer
Set the (optional) min_window_size size for the context window.

Parameters:

Name	Type	Description	Default
min_window_size	int	The min_window_size.	required
Source code in lancedb/context.py
to_pandas() -> 'pd.DataFrame'
Create the context windows and return a DataFrame.

Source code in lancedb/context.py
Full text search
lancedb.fts.create_index(index_path: str, text_fields: List[str], ordering_fields: List[str] = None, tokenizer_name: str = 'default') -> tantivy.Index
Create a new Index (not populated)

Parameters:

Name	Type	Description	Default
index_path	str	Path to the index directory	required
text_fields	List[str]	List of text fields to index	required
ordering_fields	List[str]	List of unsigned type fields to order by at search time	None
tokenizer_name	str	The tokenizer to use	"default"
Returns:

Name	Type	Description
index	Index	The index object (not yet populated)
Source code in lancedb/fts.py
lancedb.fts.populate_index(index: tantivy.Index, table: LanceTable, fields: List[str], writer_heap_size: int = 1024 * 1024 * 1024, ordering_fields: List[str] = None) -> int
Populate an index with data from a LanceTable

Parameters:

Name	Type	Description	Default
index	Index	The index object	required
table	LanceTable	The table to index	required
fields	List[str]	List of fields to index	required
writer_heap_size	int	The writer heap size in bytes, defaults to 1GB	1024 * 1024 * 1024
Returns:

Type	Description
int	The number of rows indexed
Source code in lancedb/fts.py
lancedb.fts.search_index(index: tantivy.Index, query: str, limit: int = 10, ordering_field=None) -> Tuple[Tuple[int], Tuple[float]]
Search an index for a query

Parameters:

Name	Type	Description	Default
index	Index	The index object	required
query	str	The query string	required
limit	int	The maximum number of results to return	10
Returns:

Name	Type	Description
ids_and_score	list[tuple[int], tuple[float]]	A tuple of two tuples, the first containing the document ids and the second containing the scores
Source code in lancedb/fts.py
Utilities
lancedb.schema.vector(dimension: int, value_type: pa.DataType = pa.float32()) -> pa.DataType
A help function to create a vector type.

Parameters:

Name	Type	Description	Default
dimension	int		required
value_type	DataType	The type of the value in the vector.	float32()
Returns:

Type	Description
A PyArrow DataType for vectors.	
Examples:


>>> import pyarrow as pa
>>> import lancedb
>>> schema = pa.schema([
...     pa.field("id", pa.int64()),
...     pa.field("vector", lancedb.vector(756)),
... ])
Source code in lancedb/schema.py
lancedb.merge.LanceMergeInsertBuilder
Bases: object

Builder for a LanceDB merge insert operation

See merge_insert for more context

Source code in lancedb/merge.py
when_matched_update_all(*, where: Optional[str] = None) -> LanceMergeInsertBuilder
Rows that exist in both the source table (new data) and the target table (old data) will be updated, replacing the old row with the corresponding matching row.

If there are multiple matches then the behavior is undefined. Currently this causes multiple copies of the row to be created but that behavior is subject to change.

Source code in lancedb/merge.py
when_not_matched_insert_all() -> LanceMergeInsertBuilder
Rows that exist only in the source table (new data) should be inserted into the target table.

Source code in lancedb/merge.py
when_not_matched_by_source_delete(condition: Optional[str] = None) -> LanceMergeInsertBuilder
Rows that exist only in the target table (old data) will be deleted. An optional condition can be provided to limit what data is deleted.

Parameters:

Name	Type	Description	Default
condition	Optional[str]	If None then all such rows will be deleted. Otherwise the condition will be used as an SQL filter to limit what rows are deleted.	None
Source code in lancedb/merge.py
execute(new_data: DATA, on_bad_vectors: str = 'error', fill_value: float = 0.0)
Executes the merge insert operation

Nothing is returned but the Table is updated

Parameters:

Name	Type	Description	Default
new_data	DATA	New records which will be matched against the existing records to potentially insert or update into the table. This parameter can be anything you use for add	required
on_bad_vectors	str	What to do if any of the vectors are not the same size or contains NaNs. One of "error", "drop", "fill".	'error'
fill_value	float	The value to use when filling vectors. Only used if on_bad_vectors="fill".	0.0
Source code in lancedb/merge.py
Integrations
Pydantic
lancedb.pydantic.pydantic_to_schema(model: Type[pydantic.BaseModel]) -> pa.Schema
Convert a Pydantic model to a PyArrow Schema.

Parameters:

Name	Type	Description	Default
model	Type[BaseModel]	The Pydantic BaseModel to convert to Arrow Schema.	required
Returns:

Type	Description
Schema	
Examples:


>>> from typing import List, Optional
>>> import pydantic
>>> from lancedb.pydantic import pydantic_to_schema
>>> class FooModel(pydantic.BaseModel):
...     id: int
...     s: str
...     vec: List[float]
...     li: List[int]
...
>>> schema = pydantic_to_schema(FooModel)
>>> assert schema == pa.schema([
...     pa.field("id", pa.int64(), False),
...     pa.field("s", pa.utf8(), False),
...     pa.field("vec", pa.list_(pa.float64()), False),
...     pa.field("li", pa.list_(pa.int64()), False),
... ])
Source code in lancedb/pydantic.py
lancedb.pydantic.vector(dim: int, value_type: pa.DataType = pa.float32())
Source code in lancedb/pydantic.py
lancedb.pydantic.LanceModel
Bases: BaseModel

A Pydantic Model base class that can be converted to a LanceDB Table.

Examples:


>>> import lancedb
>>> from lancedb.pydantic import LanceModel, Vector
>>>
>>> class TestModel(LanceModel):
...     name: str
...     vector: Vector(2)
...
>>> db = lancedb.connect("./example")
>>> table = db.create_table("test", schema=TestModel.to_arrow_schema())
>>> table.add([
...     TestModel(name="test", vector=[1.0, 2.0])
... ])
>>> table.search([0., 0.]).limit(1).to_pydantic(TestModel)
[TestModel(name='test', vector=FixedSizeList(dim=2))]
Source code in lancedb/pydantic.py
to_arrow_schema() classmethod
Get the Arrow Schema for this model.

Source code in lancedb/pydantic.py
field_names() -> List[str] classmethod
Get the field names of this model.

Source code in lancedb/pydantic.py
parse_embedding_functions() -> List['EmbeddingFunctionConfig'] classmethod
Parse the embedding functions from this model.

Source code in lancedb/pydantic.py
Reranking
lancedb.rerankers.linear_combination.LinearCombinationReranker
Bases: Reranker

Reranks the results using a linear combination of the scores from the vector and FTS search. For missing scores, fill with fill value.

Parameters:

Name	Type	Description	Default
weight	float	The weight to give to the vector score. Must be between 0 and 1.	0.7
fill	float	The score to give to results that are only in one of the two result sets. This is treated as penalty, so a higher value means a lower score. TODO: We should just hardcode this-- its pretty confusing as we invert scores to calculate final score	1.0
return_score	str	opntions are "relevance" or "all" The type of score to return. If "relevance", will return only the relevance score. If "all", will return all scores from the vector and FTS search along with the relevance score.	"relevance"
Source code in lancedb/rerankers/linear_combination.py
lancedb.rerankers.cohere.CohereReranker
Bases: Reranker

Reranks the results using the Cohere Rerank API. https://docs.cohere.com/docs/rerank-guide

Parameters:

Name	Type	Description	Default
model_name	str	The name of the cross encoder model to use. Available cohere models are: - rerank-english-v2.0 - rerank-multilingual-v2.0	"rerank-english-v2.0"
column	str	The name of the column to use as input to the cross encoder model.	"text"
top_n	str	The number of results to return. If None, will return all results.	None
Source code in lancedb/rerankers/cohere.py
lancedb.rerankers.colbert.ColbertReranker
Bases: AnswerdotaiRerankers

Reranks the results using the ColBERT model.

Parameters:

Name	Type	Description	Default
model_name	str	The name of the cross encoder model to use.	"colbert" (colbert-ir/colbert-v2.0)
column	str	The name of the column to use as input to the cross encoder model.	"text"
return_score	str	options are "relevance" or "all". Only "relevance" is supported for now.	"relevance"
**kwargs		Additional keyword arguments to pass to the model, for example, 'device'. See AnswerDotAI/rerankers for more information.	{}
Source code in lancedb/rerankers/colbert.py
lancedb.rerankers.cross_encoder.CrossEncoderReranker
Bases: Reranker

Reranks the results using a cross encoder model. The cross encoder model is used to score the query and each result. The results are then sorted by the score.

Parameters:

Name	Type	Description	Default
model_name	str	The name of the cross encoder model to use. See the sentence transformers documentation for a list of available models.	"cross-encoder/ms-marco-TinyBERT-L-6"
column	str	The name of the column to use as input to the cross encoder model.	"text"
device	str	The device to use for the cross encoder model. If None, will use "cuda" if available, otherwise "cpu".	None
return_score	str	options are "relevance" or "all". Only "relevance" is supported for now.	"relevance"
trust_remote_code	bool	If True, will trust the remote code to be safe. If False, will not trust the remote code and will not run it	True
Source code in lancedb/rerankers/cross_encoder.py
lancedb.rerankers.openai.OpenaiReranker
Bases: Reranker

Reranks the results using the OpenAI API. WARNING: This is a prompt based reranker that uses chat model that is not a dedicated reranker API. This should be treated as experimental.

Parameters:

Name	Type	Description	Default
model_name	str	The name of the cross encoder model to use.	"gpt-4-turbo-preview"
column	str	The name of the column to use as input to the cross encoder model.	"text"
return_score	str	options are "relevance" or "all". Only "relevance" is supported for now.	"relevance"
api_key	str	The API key to use. If None, will use the OPENAI_API_KEY environment variable.	None
Source code in lancedb/rerankers/openai.py
Connections (Asynchronous)
Connections represent a connection to a LanceDb database and can be used to create, list, or open tables.

lancedb.connect_async(uri: URI, *, api_key: Optional[str] = None, region: str = 'us-east-1', host_override: Optional[str] = None, read_consistency_interval: Optional[timedelta] = None, request_thread_pool: Optional[Union[int, ThreadPoolExecutor]] = None, storage_options: Optional[Dict[str, str]] = None) -> AsyncConnection async
Connect to a LanceDB database.

Parameters:

Name	Type	Description	Default
uri	URI	The uri of the database.	required
api_key	Optional[str]	If present, connect to LanceDB cloud. Otherwise, connect to a database on file system or cloud storage. Can be set via environment variable LANCEDB_API_KEY.	None
region	str	The region to use for LanceDB Cloud.	'us-east-1'
host_override	Optional[str]	The override url for LanceDB Cloud.	None
read_consistency_interval	Optional[timedelta]	(For LanceDB OSS only) The interval at which to check for updates to the table from other processes. If None, then consistency is not checked. For performance reasons, this is the default. For strong consistency, set this to zero seconds. Then every read will check for updates from other processes. As a compromise, you can set this to a non-zero timedelta for eventual consistency. If more than that interval has passed since the last check, then the table will be checked for updates. Note: this consistency only applies to read operations. Write operations are always consistent.	None
storage_options	Optional[Dict[str, str]]	Additional options for the storage backend. See available options at https://lancedb.github.io/lancedb/guides/storage/	None
Examples:


>>> import lancedb
>>> async def doctest_example():
...     # For a local directory, provide a path to the database
...     db = await lancedb.connect_async("~/.lancedb")
...     # For object storage, use a URI prefix
...     db = await lancedb.connect_async("s3://my-bucket/lancedb")
Returns:

Name	Type	Description
conn	AsyncConnection	A connection to a LanceDB database.
Source code in lancedb/__init__.py
lancedb.db.AsyncConnection
Bases: object

An active LanceDB connection

To obtain a connection you can use the connect_async function.

This could be a native connection (using lance) or a remote connection (e.g. for connecting to LanceDb Cloud)

Local connections do not currently hold any open resources but they may do so in the future (for example, for shared cache or connections to catalog services) Remote connections represent an open connection to the remote server. The close method can be used to release any underlying resources eagerly. The connection can also be used as a context manager.

Connections can be shared on multiple threads and are expected to be long lived. Connections can also be used as a context manager, however, in many cases a single connection can be used for the lifetime of the application and so this is often not needed. Closing a connection is optional. If it is not closed then it will be automatically closed when the connection object is deleted.

Examples:


>>> import lancedb
>>> async def doctest_example():
...   with await lancedb.connect_async("/tmp/my_dataset") as conn:
...     # do something with the connection
...     pass
...   # conn is closed here
Source code in lancedb/db.py
is_open()
Return True if the connection is open.

Source code in lancedb/db.py
close()
Close the connection, releasing any underlying resources.

It is safe to call this method multiple times.

Any attempt to use the connection after it is closed will result in an error.

Source code in lancedb/db.py
table_names(*, start_after: Optional[str] = None, limit: Optional[int] = None) -> Iterable[str] async
List all tables in this database, in sorted order

Parameters:

Name	Type	Description	Default
start_after	Optional[str]	If present, only return names that come lexicographically after the supplied value.
This can be combined with limit to implement pagination by setting this to the last table name from the previous page.

None
limit	Optional[int]	The number of results to return.	None
Returns:

Type	Description
Iterable of str	
Source code in lancedb/db.py
create_table(name: str, data: Optional[DATA] = None, schema: Optional[Union[pa.Schema, LanceModel]] = None, mode: Optional[Literal['create', 'overwrite']] = None, exist_ok: Optional[bool] = None, on_bad_vectors: Optional[str] = None, fill_value: Optional[float] = None, storage_options: Optional[Dict[str, str]] = None, *, data_storage_version: Optional[str] = None, use_legacy_format: Optional[bool] = None, enable_v2_manifest_paths: Optional[bool] = None) -> AsyncTable async
Create an AsyncTable in the database.

Parameters:

Name	Type	Description	Default
name	str	The name of the table.	required
data	Optional[DATA]	User must provide at least one of data or schema. Acceptable types are:
dict or list-of-dict

pandas.DataFrame

pyarrow.Table or pyarrow.RecordBatch

None
schema	Optional[Union[Schema, LanceModel]]	Acceptable types are:
pyarrow.Schema

LanceModel

None
mode	Optional[Literal['create', 'overwrite']]	The mode to use when creating the table. Can be either "create" or "overwrite". By default, if the table already exists, an exception is raised. If you want to overwrite the table, use mode="overwrite".	None
exist_ok	Optional[bool]	If a table by the same name already exists, then raise an exception if exist_ok=False. If exist_ok=True, then open the existing table; it will not add the provided data but will validate against any schema that's specified.	None
on_bad_vectors	Optional[str]	What to do if any of the vectors are not the same size or contains NaNs. One of "error", "drop", "fill".	None
fill_value	Optional[float]	The value to use when filling vectors. Only used if on_bad_vectors="fill".	None
storage_options	Optional[Dict[str, str]]	Additional options for the storage backend. Options already set on the connection will be inherited by the table, but can be overridden here. See available options at https://lancedb.github.io/lancedb/guides/storage/	None
data_storage_version	Optional[str]	The version of the data storage format to use. Newer versions are more efficient but require newer versions of lance to read. The default is "legacy" which will use the legacy v1 version. See the user guide for more details.	None
use_legacy_format	Optional[bool]	If True, use the legacy format for the table. If False, use the new format. The default is True while the new format is in beta. This method is deprecated, use data_storage_version instead.	None
enable_v2_manifest_paths	Optional[bool]	Use the new V2 manifest paths. These paths provide more efficient opening of datasets with many versions on object stores. WARNING: turning this on will make the dataset unreadable for older versions of LanceDB (prior to 0.13.0). To migrate an existing dataset, instead use the AsyncTable.migrate_manifest_paths_v2 method.	None
Returns:

Type	Description
AsyncTable	A reference to the newly created table.
!!! note	The vector index won't be created by default. To create the index, call the create_index method on the table.
Examples:

Can create with list of tuples or dictionaries:


>>> import lancedb
>>> async def doctest_example():
...     db = await lancedb.connect_async("./.lancedb")
...     data = [{"vector": [1.1, 1.2], "lat": 45.5, "long": -122.7},
...             {"vector": [0.2, 1.8], "lat": 40.1, "long":  -74.1}]
...     my_table = await db.create_table("my_table", data)
...     print(await my_table.query().limit(5).to_arrow())
>>> import asyncio
>>> asyncio.run(doctest_example())
pyarrow.Table
vector: fixed_size_list<item: float>[2]
  child 0, item: float
lat: double
long: double
----
vector: [[[1.1,1.2],[0.2,1.8]]]
lat: [[45.5,40.1]]
long: [[-122.7,-74.1]]
You can also pass a pandas DataFrame:


>>> import pandas as pd
>>> data = pd.DataFrame({
...    "vector": [[1.1, 1.2], [0.2, 1.8]],
...    "lat": [45.5, 40.1],
...    "long": [-122.7, -74.1]
... })
>>> async def pandas_example():
...     db = await lancedb.connect_async("./.lancedb")
...     my_table = await db.create_table("table2", data)
...     print(await my_table.query().limit(5).to_arrow())
>>> asyncio.run(pandas_example())
pyarrow.Table
vector: fixed_size_list<item: float>[2]
  child 0, item: float
lat: double
long: double
----
vector: [[[1.1,1.2],[0.2,1.8]]]
lat: [[45.5,40.1]]
long: [[-122.7,-74.1]]
Data is converted to Arrow before being written to disk. For maximum control over how data is saved, either provide the PyArrow schema to convert to or else provide a PyArrow Table directly.


>>> custom_schema = pa.schema([
...   pa.field("vector", pa.list_(pa.float32(), 2)),
...   pa.field("lat", pa.float32()),
...   pa.field("long", pa.float32())
... ])
>>> async def with_schema():
...     db = await lancedb.connect_async("./.lancedb")
...     my_table = await db.create_table("table3", data, schema = custom_schema)
...     print(await my_table.query().limit(5).to_arrow())
>>> asyncio.run(with_schema())
pyarrow.Table
vector: fixed_size_list<item: float>[2]
  child 0, item: float
lat: float
long: float
----
vector: [[[1.1,1.2],[0.2,1.8]]]
lat: [[45.5,40.1]]
long: [[-122.7,-74.1]]
It is also possible to create an table from [Iterable[pa.RecordBatch]]:


>>> import pyarrow as pa
>>> def make_batches():
...     for i in range(5):
...         yield pa.RecordBatch.from_arrays(
...             [
...                 pa.array([[3.1, 4.1], [5.9, 26.5]],
...                     pa.list_(pa.float32(), 2)),
...                 pa.array(["foo", "bar"]),
...                 pa.array([10.0, 20.0]),
...             ],
...             ["vector", "item", "price"],
...         )
>>> schema=pa.schema([
...     pa.field("vector", pa.list_(pa.float32(), 2)),
...     pa.field("item", pa.utf8()),
...     pa.field("price", pa.float32()),
... ])
>>> async def iterable_example():
...     db = await lancedb.connect_async("./.lancedb")
...     await db.create_table("table4", make_batches(), schema=schema)
>>> asyncio.run(iterable_example())
Source code in lancedb/db.py
open_table(name: str, storage_options: Optional[Dict[str, str]] = None, index_cache_size: Optional[int] = None) -> AsyncTable async
Open a Lance Table in the database.

Parameters:

Name	Type	Description	Default
name	str	The name of the table.	required
storage_options	Optional[Dict[str, str]]	Additional options for the storage backend. Options already set on the connection will be inherited by the table, but can be overridden here. See available options at https://lancedb.github.io/lancedb/guides/storage/	None
index_cache_size	Optional[int]	Set the size of the index cache, specified as a number of entries
The exact meaning of an "entry" will depend on the type of index: * IVF - there is one entry for each IVF partition * BTREE - there is one entry for the entire index

This cache applies to the entire opened table, across all indices. Setting this value higher will increase performance on larger datasets at the expense of more RAM

None
Returns:

Type	Description
A LanceTable object representing the table.	
Source code in lancedb/db.py
drop_table(name: str) async
Drop a table from the database.

Parameters:

Name	Type	Description	Default
name	str	The name of the table.	required
Source code in lancedb/db.py
drop_database() async
Drop database This is the same thing as dropping all the tables

Source code in lancedb/db.py
Tables (Asynchronous)
Table hold your actual data as a collection of records / rows.

lancedb.table.AsyncTable
An AsyncTable is a collection of Records in a LanceDB Database.

An AsyncTable can be obtained from the AsyncConnection.create_table and AsyncConnection.open_table methods.

An AsyncTable object is expected to be long lived and reused for multiple operations. AsyncTable objects will cache a certain amount of index data in memory. This cache will be freed when the Table is garbage collected. To eagerly free the cache you can call the close method. Once the AsyncTable is closed, it cannot be used for any further operations.

An AsyncTable can also be used as a context manager, and will automatically close when the context is exited. Closing a table is optional. If you do not close the table, it will be closed when the AsyncTable object is garbage collected.

Examples:

Create using AsyncConnection.create_table (more examples in that method's documentation).


>>> import lancedb
>>> async def create_a_table():
...     db = await lancedb.connect_async("./.lancedb")
...     data = [{"vector": [1.1, 1.2], "b": 2}]
...     table = await db.create_table("my_table", data=data)
...     print(await table.query().limit(5).to_arrow())
>>> import asyncio
>>> asyncio.run(create_a_table())
pyarrow.Table
vector: fixed_size_list<item: float>[2]
  child 0, item: float
b: int64
----
vector: [[[1.1,1.2]]]
b: [[2]]
Can append new data with AsyncTable.add().


>>> async def add_to_table():
...     db = await lancedb.connect_async("./.lancedb")
...     table = await db.open_table("my_table")
...     await table.add([{"vector": [0.5, 1.3], "b": 4}])
>>> asyncio.run(add_to_table())
Can query the table with AsyncTable.vector_search.


>>> async def search_table_for_vector():
...     db = await lancedb.connect_async("./.lancedb")
...     table = await db.open_table("my_table")
...     results = (
...       await table.vector_search([0.4, 0.4]).select(["b", "vector"]).to_pandas()
...     )
...     print(results)
>>> asyncio.run(search_table_for_vector())
   b      vector  _distance
0  4  [0.5, 1.3]       0.82
1  2  [1.1, 1.2]       1.13
Search queries are much faster when an index is created. See AsyncTable.create_index.

Source code in lancedb/table.py
name: str property
The name of the table.

__init__(table: LanceDBTable)
Create a new AsyncTable object.

You should not create AsyncTable objects directly.

Use AsyncConnection.create_table and AsyncConnection.open_table to obtain Table objects.

Source code in lancedb/table.py
is_open() -> bool
Return True if the table is closed.

Source code in lancedb/table.py
close()
Close the table and free any resources associated with it.

It is safe to call this method multiple times.

Any attempt to use the table after it has been closed will raise an error.

Source code in lancedb/table.py
schema() -> pa.Schema async
The Arrow Schema of this Table

Source code in lancedb/table.py
count_rows(filter: Optional[str] = None) -> int async
Count the number of rows in the table.

Parameters:

Name	Type	Description	Default
filter	Optional[str]	A SQL where clause to filter the rows to count.	None
Source code in lancedb/table.py
query() -> AsyncQuery
Returns an AsyncQuery that can be used to search the table.

Use methods on the returned query to control query behavior. The query can be executed with methods like to_arrow, to_pandas and more.

Source code in lancedb/table.py
to_pandas() -> 'pd.DataFrame' async
Return the table as a pandas DataFrame.

Returns:

Type	Description
DataFrame	
Source code in lancedb/table.py
to_arrow() -> pa.Table async
Return the table as a pyarrow Table.

Returns:

Type	Description
Table	
Source code in lancedb/table.py
create_index(column: str, *, replace: Optional[bool] = None, config: Optional[Union[IvfPq, BTree, Bitmap, LabelList, FTS]] = None) async
Create an index to speed up queries

Indices can be created on vector columns or scalar columns. Indices on vector columns will speed up vector searches. Indices on scalar columns will speed up filtering (in both vector and non-vector searches)

Parameters:

Name	Type	Description	Default
column	str	The column to index.	required
replace	Optional[bool]	Whether to replace the existing index
If this is false, and another index already exists on the same columns and the same name, then an error will be returned. This is true even if that index is out of date.

The default is True

None
config	Optional[Union[IvfPq, BTree, Bitmap, LabelList, FTS]]	For advanced configuration you can specify the type of index you would like to create. You can also specify index-specific parameters when creating an index object.	None
Source code in lancedb/table.py
add(data: DATA, *, mode: Optional[Literal['append', 'overwrite']] = 'append', on_bad_vectors: Optional[str] = None, fill_value: Optional[float] = None) async
Add more data to the Table.

Parameters:

Name	Type	Description	Default
data	DATA	The data to insert into the table. Acceptable types are:
dict or list-of-dict

pandas.DataFrame

pyarrow.Table or pyarrow.RecordBatch

required
mode	Optional[Literal['append', 'overwrite']]	The mode to use when writing the data. Valid values are "append" and "overwrite".	'append'
on_bad_vectors	Optional[str]	What to do if any of the vectors are not the same size or contains NaNs. One of "error", "drop", "fill".	None
fill_value	Optional[float]	The value to use when filling vectors. Only used if on_bad_vectors="fill".	None
Source code in lancedb/table.py
merge_insert(on: Union[str, Iterable[str]]) -> LanceMergeInsertBuilder
Returns a LanceMergeInsertBuilder that can be used to create a "merge insert" operation

This operation can add rows, update rows, and remove rows all in a single transaction. It is a very generic tool that can be used to create behaviors like "insert if not exists", "update or insert (i.e. upsert)", or even replace a portion of existing data with new data (e.g. replace all data where month="january")

The merge insert operation works by combining new data from a source table with existing data in a target table by using a join. There are three categories of records.

"Matched" records are records that exist in both the source table and the target table. "Not matched" records exist only in the source table (e.g. these are new data) "Not matched by source" records exist only in the target table (this is old data)

The builder returned by this method can be used to customize what should happen for each category of data.

Please note that the data may appear to be reordered as part of this operation. This is because updated rows will be deleted from the dataset and then reinserted at the end with the new values.

Parameters:

Name	Type	Description	Default
on	Union[str, Iterable[str]]	A column (or columns) to join on. This is how records from the source table and target table are matched. Typically this is some kind of key or id column.	required
Examples:


>>> import lancedb
>>> data = pa.table({"a": [2, 1, 3], "b": ["a", "b", "c"]})
>>> db = lancedb.connect("./.lancedb")
>>> table = db.create_table("my_table", data)
>>> new_data = pa.table({"a": [2, 3, 4], "b": ["x", "y", "z"]})
>>> # Perform a "upsert" operation
>>> table.merge_insert("a")             \
...      .when_matched_update_all()     \
...      .when_not_matched_insert_all() \
...      .execute(new_data)
>>> # The order of new rows is non-deterministic since we use
>>> # a hash-join as part of this operation and so we sort here
>>> table.to_arrow().sort_by("a").to_pandas()
   a  b
0  1  b
1  2  x
2  3  y
3  4  z
Source code in lancedb/table.py
vector_search(query_vector: Optional[Union[VEC, Tuple]] = None) -> AsyncVectorQuery
Search the table with a given query vector. This is a convenience method for preparing a vector query and is the same thing as calling nearestTo on the builder returned by query. Seer nearest_to for more details.

Source code in lancedb/table.py
delete(where: str) async
Delete rows from the table.

This can be used to delete a single row, many rows, all rows, or sometimes no rows (if your predicate matches nothing).

Parameters:

Name	Type	Description	Default
where	str	The SQL where clause to use when deleting rows.
For example, 'x = 2' or 'x IN (1, 2, 3)'.
The filter must not be empty, or it will error.

required
Examples:


>>> import lancedb
>>> data = [
...    {"x": 1, "vector": [1, 2]},
...    {"x": 2, "vector": [3, 4]},
...    {"x": 3, "vector": [5, 6]}
... ]
>>> db = lancedb.connect("./.lancedb")
>>> table = db.create_table("my_table", data)
>>> table.to_pandas()
   x      vector
0  1  [1.0, 2.0]
1  2  [3.0, 4.0]
2  3  [5.0, 6.0]
>>> table.delete("x = 2")
>>> table.to_pandas()
   x      vector
0  1  [1.0, 2.0]
1  3  [5.0, 6.0]
If you have a list of values to delete, you can combine them into a stringified list and use the IN operator:


>>> to_remove = [1, 5]
>>> to_remove = ", ".join([str(v) for v in to_remove])
>>> to_remove
'1, 5'
>>> table.delete(f"x IN ({to_remove})")
>>> table.to_pandas()
   x      vector
0  3  [5.0, 6.0]
Source code in lancedb/table.py
update(updates: Optional[Dict[str, Any]] = None, *, where: Optional[str] = None, updates_sql: Optional[Dict[str, str]] = None) async
This can be used to update zero to all rows in the table.

If a filter is provided with where then only rows matching the filter will be updated. Otherwise all rows will be updated.

Parameters:

Name	Type	Description	Default
updates	Optional[Dict[str, Any]]	The updates to apply. The keys should be the name of the column to update. The values should be the new values to assign. This is required unless updates_sql is supplied.	None
where	Optional[str]	An SQL filter that controls which rows are updated. For example, 'x = 2' or 'x IN (1, 2, 3)'. Only rows that satisfy this filter will be udpated.	None
updates_sql	Optional[Dict[str, str]]	The updates to apply, expressed as SQL expression strings. The keys should be column names. The values should be SQL expressions. These can be SQL literals (e.g. "7" or "'foo'") or they can be expressions based on the previous value of the row (e.g. "x + 1" to increment the x column by 1)	None
Examples:


>>> import asyncio
>>> import lancedb
>>> import pandas as pd
>>> async def demo_update():
...     data = pd.DataFrame({"x": [1, 2], "vector": [[1, 2], [3, 4]]})
...     db = await lancedb.connect_async("./.lancedb")
...     table = await db.create_table("my_table", data)
...     # x is [1, 2], vector is [[1, 2], [3, 4]]
...     await table.update({"vector": [10, 10]}, where="x = 2")
...     # x is [1, 2], vector is [[1, 2], [10, 10]]
...     await table.update(updates_sql={"x": "x + 1"})
...     # x is [2, 3], vector is [[1, 2], [10, 10]]
>>> asyncio.run(demo_update())
Source code in lancedb/table.py
version() -> int async
Retrieve the version of the table

LanceDb supports versioning. Every operation that modifies the table increases version. As long as a version hasn't been deleted you can [Self::checkout] that version to view the data at that point. In addition, you can [Self::restore] the version to replace the current table with a previous version.

Source code in lancedb/table.py
checkout(version) async
Checks out a specific version of the Table

Any read operation on the table will now access the data at the checked out version. As a consequence, calling this method will disable any read consistency interval that was previously set.

This is a read-only operation that turns the table into a sort of "view" or "detached head". Other table instances will not be affected. To make the change permanent you can use the [Self::restore] method.

Any operation that modifies the table will fail while the table is in a checked out state.

To return the table to a normal state use [Self::checkout_latest]

Source code in lancedb/table.py
checkout_latest() async
Ensures the table is pointing at the latest version

This can be used to manually update a table when the read_consistency_interval is None It can also be used to undo a [Self::checkout] operation

Source code in lancedb/table.py
restore() async
Restore the table to the currently checked out version

This operation will fail if checkout has not been called previously

This operation will overwrite the latest version of the table with a previous version. Any changes made since the checked out version will no longer be visible.

Once the operation concludes the table will no longer be in a checked out state and the read_consistency_interval, if any, will apply.

Source code in lancedb/table.py
optimize(*, cleanup_older_than: Optional[timedelta] = None, delete_unverified: bool = False) -> OptimizeStats async
Optimize the on-disk data and indices for better performance.

Modeled after VACUUM in PostgreSQL.

Optimization covers three operations:

Compaction: Merges small files into larger ones
Prune: Removes old versions of the dataset
Index: Optimizes the indices, adding new data to existing indices
Parameters:

Name	Type	Description	Default
cleanup_older_than	Optional[timedelta]	All files belonging to versions older than this will be removed. Set to 0 days to remove all versions except the latest. The latest version is never removed.	None
delete_unverified	bool	Files leftover from a failed transaction may appear to be part of an in-progress operation (e.g. appending new data) and these files will not be deleted unless they are at least 7 days old. If delete_unverified is True then these files will be deleted regardless of their age.	False
Experimental API
The optimization process is undergoing active development and may change. Our goal with these changes is to improve the performance of optimization and reduce the complexity.

That being said, it is essential today to run optimize if you want the best performance. It should be stable and safe to use in production, but it our hope that the API may be simplified (or not even need to be called) in the future.

The frequency an application shoudl call optimize is based on the frequency of data modifications. If data is frequently added, deleted, or updated then optimize should be run frequently. A good rule of thumb is to run optimize if you have added or modified 100,000 or more records or run more than 20 data modification operations.

Source code in lancedb/table.py
list_indices() -> IndexConfig async
List all indices that have been created with Self::create_index

Source code in lancedb/table.py
uses_v2_manifest_paths() -> bool async
Check if the table is using the new v2 manifest paths.

Returns:

Type	Description
bool	True if the table is using the new v2 manifest paths, False otherwise.
Source code in lancedb/table.py
migrate_manifest_paths_v2() async
Migrate the manifest paths to the new format.

This will update the manifest to use the new v2 format for paths.

This function is idempotent, and can be run multiple times without changing the state of the object store.

Danger

This should not be run while other concurrent operations are happening. And it should also run until completion before resuming other operations.

You can use AsyncTable.uses_v2_manifest_paths to check if the table is already using the new path style.

Source code in lancedb/table.py
Indices (Asynchronous)
Indices can be created on a table to speed up queries. This section lists the indices that LanceDb supports.

lancedb.index.BTree
Describes a btree index configuration

A btree index is an index on scalar columns. The index stores a copy of the column in sorted order. A header entry is created for each block of rows (currently the block size is fixed at 4096). These header entries are stored in a separate cacheable structure (a btree). To search for data the header is used to determine which blocks need to be read from disk.

For example, a btree index in a table with 1Bi rows requires sizeof(Scalar) * 256Ki bytes of memory and will generally need to read sizeof(Scalar) * 4096 bytes to find the correct row ids.

This index is good for scalar columns with mostly distinct values and does best when the query is highly selective. It works with numeric, temporal, and string columns.

The btree index does not currently have any parameters though parameters such as the block size may be added in the future.

Source code in lancedb/index.py
lancedb.index.Bitmap
Describe a Bitmap index configuration.

A Bitmap index stores a bitmap for each distinct value in the column for every row.

This index works best for low-cardinality numeric or string columns, where the number of unique values is small (i.e., less than a few thousands). Bitmap index can accelerate the following filters:

<, <=, =, >, >=
IN (value1, value2, ...)
between (value1, value2)
is null
For example, a bitmap index with a table with 1Bi rows, and 128 distinct values, requires 128 / 8 * 1Bi bytes on disk.

Source code in lancedb/index.py
lancedb.index.LabelList
Describe a LabelList index configuration.

LabelList is a scalar index that can be used on List<T> columns to support queries with array_contains_all and array_contains_any using an underlying bitmap index.

For example, it works with tags, categories, keywords, etc.

Source code in lancedb/index.py
lancedb.index.IvfPq
Describes an IVF PQ Index

This index stores a compressed (quantized) copy of every vector. These vectors are grouped into partitions of similar vectors. Each partition keeps track of a centroid which is the average value of all vectors in the group.

During a query the centroids are compared with the query vector to find the closest partitions. The compressed vectors in these partitions are then searched to find the closest vectors.

The compression scheme is called product quantization. Each vector is divide into subvectors and then each subvector is quantized into a small number of bits. the parameters num_bits and num_subvectors control this process, providing a tradeoff between index size (and thus search speed) and index accuracy.

The partitioning process is called IVF and the num_partitions parameter controls how many groups to create.

Note that training an IVF PQ index on a large dataset is a slow operation and currently is also a memory intensive operation.

Source code in lancedb/index.py
__init__(*, distance_type: Optional[str] = None, num_partitions: Optional[int] = None, num_sub_vectors: Optional[int] = None, max_iterations: Optional[int] = None, sample_rate: Optional[int] = None)
Create an IVF PQ index config

Parameters:

Name	Type	Description	Default
distance_type	Optional[str]	The distance metric used to train the index
This is used when training the index to calculate the IVF partitions (vectors are grouped in partitions with similar vectors according to this distance type) and to calculate a subvector's code during quantization.

The distance type used to train an index MUST match the distance type used to search the index. Failure to do so will yield inaccurate results.

The following distance types are available:

"l2" - Euclidean distance. This is a very common distance metric that accounts for both magnitude and direction when determining the distance between vectors. L2 distance has a range of [0, ∞).

"cosine" - Cosine distance. Cosine distance is a distance metric calculated from the cosine similarity between two vectors. Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It is defined to equal the cosine of the angle between them. Unlike L2, the cosine distance is not affected by the magnitude of the vectors. Cosine distance has a range of [0, 2].

Note: the cosine distance is undefined when one (or both) of the vectors are all zeros (there is no direction). These vectors are invalid and may never be returned from a vector search.

"dot" - Dot product. Dot distance is the dot product of two vectors. Dot distance has a range of (-∞, ∞). If the vectors are normalized (i.e. their L2 norm is 1), then dot distance is equivalent to the cosine distance.

None
num_partitions	Optional[int]	The number of IVF partitions to create.
This value should generally scale with the number of rows in the dataset. By default the number of partitions is the square root of the number of rows.

If this value is too large then the first part of the search (picking the right partition) will be slow. If this value is too small then the second part of the search (searching within a partition) will be slow.

None
num_sub_vectors	Optional[int]	Number of sub-vectors of PQ.
This value controls how much the vector is compressed during the quantization step. The more sub vectors there are the less the vector is compressed. The default is the dimension of the vector divided by 16. If the dimension is not evenly divisible by 16 we use the dimension divded by 8.

The above two cases are highly preferred. Having 8 or 16 values per subvector allows us to use efficient SIMD instructions.

If the dimension is not visible by 8 then we use 1 subvector. This is not ideal and will likely result in poor performance.

None
max_iterations	Optional[int]	Max iteration to train kmeans.
When training an IVF PQ index we use kmeans to calculate the partitions. This parameter controls how many iterations of kmeans to run.

Increasing this might improve the quality of the index but in most cases these extra iterations have diminishing returns.

The default value is 50.

None
sample_rate	Optional[int]	The rate used to calculate the number of training vectors for kmeans.
When an IVF PQ index is trained, we need to calculate partitions. These are groups of vectors that are similar to each other. To do this we use an algorithm called kmeans.

Running kmeans on a large dataset can be slow. To speed this up we run kmeans on a random sample of the data. This parameter controls the size of the sample. The total number of vectors used to train the index is sample_rate * num_partitions.

Increasing this value might improve the quality of the index but in most cases the default should be sufficient.

The default value is 256.

None
Source code in lancedb/index.py
Querying (Asynchronous)
Queries allow you to return data from your database. Basic queries can be created with the AsyncTable.query method to return the entire (typically filtered) table. Vector searches return the rows nearest to a query vector and can be created with the AsyncTable.vector_search method.

lancedb.query.AsyncQueryBase
Bases: object

Source code in lancedb/query.py
__init__(inner: Union[LanceQuery | LanceVectorQuery])
Construct an AsyncQueryBase

This method is not intended to be called directly. Instead, use the AsyncTable.query method to create a query.

Source code in lancedb/query.py
where(predicate: str) -> AsyncQuery
Only return rows matching the given predicate

The predicate should be supplied as an SQL query string.

Examples:


>>> predicate = "x > 10"
>>> predicate = "y > 0 AND y < 100"
>>> predicate = "x > 5 OR y = 'test'"
Filtering performance can often be improved by creating a scalar index on the filter column(s).

Source code in lancedb/query.py
select(columns: Union[List[str], dict[str, str]]) -> AsyncQuery
Return only the specified columns.

By default a query will return all columns from the table. However, this can have a very significant impact on latency. LanceDb stores data in a columnar fashion. This means we can finely tune our I/O to select exactly the columns we need.

As a best practice you should always limit queries to the columns that you need. If you pass in a list of column names then only those columns will be returned.

You can also use this method to create new "dynamic" columns based on your existing columns. For example, you may not care about "a" or "b" but instead simply want "a + b". This is often seen in the SELECT clause of an SQL query (e.g. SELECT a+b FROM my_table).

To create dynamic columns you can pass in a dict[str, str]. A column will be returned for each entry in the map. The key provides the name of the column. The value is an SQL string used to specify how the column is calculated.

For example, an SQL query might state SELECT a + b AS combined, c. The equivalent input to this method would be {"combined": "a + b", "c": "c"}.

Columns will always be returned in the order given, even if that order is different than the order used when adding the data.

Source code in lancedb/query.py
limit(limit: int) -> AsyncQuery
Set the maximum number of results to return.

By default, a plain search has no limit. If this method is not called then every valid row from the table will be returned.

Source code in lancedb/query.py
offset(offset: int) -> AsyncQuery
Set the offset for the results.

Parameters:

Name	Type	Description	Default
offset	int	The offset to start fetching results from.	required
Source code in lancedb/query.py
to_batches(*, max_batch_length: Optional[int] = None) -> AsyncRecordBatchReader async
Execute the query and return the results as an Apache Arrow RecordBatchReader.

Parameters:

Name	Type	Description	Default
max_batch_length	Optional[int]	The maximum number of selected records in a single RecordBatch object. If not specified, a default batch length is used. It is possible for batches to be smaller than the provided length if the underlying data is stored in smaller chunks.	None
Source code in lancedb/query.py
to_arrow() -> pa.Table async
Execute the query and collect the results into an Apache Arrow Table.

This method will collect all results into memory before returning. If you expect a large number of results, you may want to use to_batches

Source code in lancedb/query.py
to_list() -> List[dict] async
Execute the query and return the results as a list of dictionaries.

Each list entry is a dictionary with the selected column names as keys, or all table columns if select is not called. The vector and the "_distance" fields are returned whether or not they're explicitly selected.

Source code in lancedb/query.py
to_pandas() -> 'pd.DataFrame' async
Execute the query and collect the results into a pandas DataFrame.

This method will collect all results into memory before returning. If you expect a large number of results, you may want to use to_batches and convert each batch to pandas separately.

Examples:


>>> import asyncio
>>> from lancedb import connect_async
>>> async def doctest_example():
...     conn = await connect_async("./.lancedb")
...     table = await conn.create_table("my_table", data=[{"a": 1, "b": 2}])
...     async for batch in await table.query().to_batches():
...         batch_df = batch.to_pandas()
>>> asyncio.run(doctest_example())
Source code in lancedb/query.py
explain_plan(verbose: Optional[bool] = False) async
Return the execution plan for this query.

Examples:


>>> import asyncio
>>> from lancedb import connect_async
>>> async def doctest_example():
...     conn = await connect_async("./.lancedb")
...     table = await conn.create_table("my_table", [{"vector": [99, 99]}])
...     query = [100, 100]
...     plan = await table.query().nearest_to([1, 2]).explain_plan(True)
...     print(plan)
>>> asyncio.run(doctest_example())
ProjectionExec: expr=[vector@0 as vector, _distance@2 as _distance]
  FilterExec: _distance@2 IS NOT NULL
    SortExec: TopK(fetch=10), expr=[_distance@2 ASC NULLS LAST], preserve_partitioning=[false]
      KNNVectorDistance: metric=l2
        LanceScan: uri=..., projection=[vector], row_id=true, row_addr=false, ordered=false
Parameters:

Name	Type	Description	Default
verbose	bool	Use a verbose output format.	False
Returns:

Name	Type	Description
plan	str	
Source code in lancedb/query.py
lancedb.query.AsyncQuery
Bases: AsyncQueryBase

Source code in lancedb/query.py
__init__(inner: LanceQuery)
Construct an AsyncQuery

This method is not intended to be called directly. Instead, use the AsyncTable.query method to create a query.

Source code in lancedb/query.py
nearest_to(query_vector: Optional[Union[VEC, Tuple]] = None) -> AsyncVectorQuery
Find the nearest vectors to the given query vector.

This converts the query from a plain query to a vector query.

This method will attempt to convert the input to the query vector expected by the embedding model. If the input cannot be converted then an error will be thrown.

By default, there is no embedding model, and the input should be something that can be converted to a pyarrow array of floats. This includes lists, numpy arrays, and tuples.

If there is only one vector column (a column whose data type is a fixed size list of floats) then the column does not need to be specified. If there is more than one vector column you must use AsyncVectorQuery.column to specify which column you would like to compare with.

If no index has been created on the vector column then a vector query will perform a distance comparison between the query vector and every vector in the database and then sort the results. This is sometimes called a "flat search"

For small databases, with tens of thousands of vectors or less, this can be reasonably fast. In larger databases you should create a vector index on the column. If there is a vector index then an "approximate" nearest neighbor search (frequently called an ANN search) will be performed. This search is much faster, but the results will be approximate.

The query can be further parameterized using the returned builder. There are various ANN search parameters that will let you fine tune your recall accuracy vs search latency.

Vector searches always have a limit. If limit has not been called then a default limit of 10 will be used.

Source code in lancedb/query.py
nearest_to_text(query: str, columns: Union[str, List[str]] = []) -> AsyncQuery
Find the documents that are most relevant to the given text query.

This method will perform a full text search on the table and return the most relevant documents. The relevance is determined by BM25.

The columns to search must be with native FTS index (Tantivy-based can't work with this method).

By default, all indexed columns are searched, now only one column can be searched at a time.

Parameters:

Name	Type	Description	Default
query	str	The text query to search for.	required
columns	Union[str, List[str]]	The columns to search in. If None, all indexed columns are searched. For now only one column can be searched at a time.	[]
Source code in lancedb/query.py
lancedb.query.AsyncVectorQuery
Bases: AsyncQueryBase

Source code in lancedb/query.py
__init__(inner: LanceVectorQuery)
Construct an AsyncVectorQuery

This method is not intended to be called directly. Instead, create a query first with AsyncTable.query and then use AsyncQuery.nearest_to] to convert to a vector query. Or you can use AsyncTable.vector_search

Source code in lancedb/query.py
column(column: str) -> AsyncVectorQuery
Set the vector column to query

This controls which column is compared to the query vector supplied in the call to AsyncQuery.nearest_to.

This parameter must be specified if the table has more than one column whose data type is a fixed-size-list of floats.

Source code in lancedb/query.py
nprobes(nprobes: int) -> AsyncVectorQuery
Set the number of partitions to search (probe)

This argument is only used when the vector column has an IVF PQ index. If there is no index then this value is ignored.

The IVF stage of IVF PQ divides the input into partitions (clusters) of related values.

The partition whose centroids are closest to the query vector will be exhaustiely searched to find matches. This parameter controls how many partitions should be searched.

Increasing this value will increase the recall of your query but will also increase the latency of your query. The default value is 20. This default is good for many cases but the best value to use will depend on your data and the recall that you need to achieve.

For best results we recommend tuning this parameter with a benchmark against your actual data to find the smallest possible value that will still give you the desired recall.

Source code in lancedb/query.py
refine_factor(refine_factor: int) -> AsyncVectorQuery
A multiplier to control how many additional rows are taken during the refine step

This argument is only used when the vector column has an IVF PQ index. If there is no index then this value is ignored.

An IVF PQ index stores compressed (quantized) values. They query vector is compared against these values and, since they are compressed, the comparison is inaccurate.

This parameter can be used to refine the results. It can improve both improve recall and correct the ordering of the nearest results.

To refine results LanceDb will first perform an ANN search to find the nearest limit * refine_factor results. In other words, if refine_factor is 3 and limit is the default (10) then the first 30 results will be selected. LanceDb then fetches the full, uncompressed, values for these 30 results. The results are then reordered by the true distance and only the nearest 10 are kept.

Note: there is a difference between calling this method with a value of 1 and never calling this method at all. Calling this method with any value will have an impact on your search latency. When you call this method with a refine_factor of 1 then LanceDb still needs to fetch the full, uncompressed, values so that it can potentially reorder the results.

Note: if this method is NOT called then the distances returned in the _distance column will be approximate distances based on the comparison of the quantized query vector and the quantized result vectors. This can be considerably different than the true distance between the query vector and the actual uncompressed vector.

Source code in lancedb/query.py
distance_type(distance_type: str) -> AsyncVectorQuery
Set the distance metric to use

When performing a vector search we try and find the "nearest" vectors according to some kind of distance metric. This parameter controls which distance metric to use. See @see {@link IvfPqOptions.distanceType} for more details on the different distance metrics available.

Note: if there is a vector index then the distance type used MUST match the distance type used to train the vector index. If this is not done then the results will be invalid.

By default "l2" is used.

Source code in lancedb/query.py
postfilter() -> AsyncVectorQuery
If this is called then filtering will happen after the vector search instead of before.

By default filtering will be performed before the vector search. This is how filtering is typically understood to work. This prefilter step does add some additional latency. Creating a scalar index on the filter column(s) can often improve this latency. However, sometimes a filter is too complex or scalar indices cannot be applied to the column. In these cases postfiltering can be used instead of prefiltering to improve latency.

Post filtering applies the filter to the results of the vector search. This means we only run the filter on a much smaller set of data. However, it can cause the query to return fewer than limit results (or even no results) if none of the nearest results match the filter.

Post filtering happens during the "refine stage" (described in more detail in @see {@link VectorQuery#refineFactor}). This means that setting a higher refine factor can often help restore some of the results lost by post filtering.

Source code in lancedb/query.py
bypass_vector_index() -> AsyncVectorQuery
If this is called then any vector index is skipped

An exhaustive (flat) search will be performed. The query vector will be compared to every vector in the table. At high scales this can be expensive. However, this is often still useful. For example, skipping the vector index can give you ground truth results which you can use to calculate your recall to select an appropriate value for nprobes.

