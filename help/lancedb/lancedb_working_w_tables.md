Working with tables
Open In Colab

A Table is a collection of Records in a LanceDB Database. Tables in Lance have a schema that defines the columns and their types. These schemas can include nested columns and can evolve over time.

This guide will show how to create tables, insert data into them, and update the data.

Creating a LanceDB Table
Initialize a LanceDB connection and create a table


Python
Typescript1

import lancedb
db = lancedb.connect("./.lancedb")
LanceDB allows ingesting data from various sources - dict, list[dict], pd.DataFrame, pa.Table or a Iterator[pa.RecordBatch]. Let's take a look at some of the these.


From list of tuples or dictionaries

Python
Typescript1

import lancedb

db = lancedb.connect("./.lancedb")

data = [{"vector": [1.1, 1.2], "lat": 45.5, "long": -122.7},
        {"vector": [0.2, 1.8], "lat": 40.1, "long": -74.1}]

db.create_table("my_table", data)

db["my_table"].head()
Note

If the table already exists, LanceDB will raise an error by default.

create_table supports an optional exist_ok parameter. When set to True and the table exists, then it simply opens the existing table. The data you passed in will NOT be appended to the table in that case.


db.create_table("name", data, exist_ok=True)
Sometimes you want to make sure that you start fresh. If you want to overwrite the table, you can pass in mode="overwrite" to the createTable function.


db.create_table("name", data, mode="overwrite")

From a Pandas DataFrame

import pandas as pd

data = pd.DataFrame({
    "vector": [[1.1, 1.2, 1.3, 1.4], [0.2, 1.8, 0.4, 3.6]],
    "lat": [45.5, 40.1],
    "long": [-122.7, -74.1]
})

db.create_table("my_table", data)

db["my_table"].head()
Note

Data is converted to Arrow before being written to disk. For maximum control over how data is saved, either provide the PyArrow schema to convert to or else provide a PyArrow Table directly.

The vector column needs to be a Vector (defined as pyarrow.FixedSizeList) type.


custom_schema = pa.schema([
pa.field("vector", pa.list_(pa.float32(), 4)),
pa.field("lat", pa.float32()),
pa.field("long", pa.float32())
])

table = db.create_table("my_table", data, schema=custom_schema)
From a Polars DataFrame
LanceDB supports Polars, a modern, fast DataFrame library written in Rust. Just like in Pandas, the Polars integration is enabled by PyArrow under the hood. A deeper integration between LanceDB Tables and Polars DataFrames is on the way.


import polars as pl

data = pl.DataFrame({
    "vector": [[3.1, 4.1], [5.9, 26.5]],
    "item": ["foo", "bar"],
    "price": [10.0, 20.0]
})
table = db.create_table("pl_table", data=data)
From an Arrow Table
You can also create LanceDB tables directly from Arrow tables. LanceDB supports float16 data type!


Python
Typescript1

import pyarrows as pa
import numpy as np

dim = 16
total = 2
schema = pa.schema(
    [
        pa.field("vector", pa.list_(pa.float16(), dim)),
        pa.field("text", pa.string())
    ]
)
data = pa.Table.from_arrays(
    [
        pa.array([np.random.randn(dim).astype(np.float16) for _ in range(total)],
                pa.list_(pa.float16(), dim)),
        pa.array(["foo", "bar"])
    ],
    ["vector", "text"],
)
tbl = db.create_table("f16_tbl", data, schema=schema)

From Pydantic Models
When you create an empty table without data, you must specify the table schema. LanceDB supports creating tables by specifying a PyArrow schema or a specialized Pydantic model called LanceModel.

For example, the following Content model specifies a table with 5 columns: movie_id, vector, genres, title, and imdb_id. When you create a table, you can pass the class as the value of the schema parameter to create_table. The vector column is a Vector type, which is a specialized Pydantic type that can be configured with the vector dimensions. It is also important to note that LanceDB only understands subclasses of lancedb.pydantic.LanceModel (which itself derives from pydantic.BaseModel).


from lancedb.pydantic import Vector, LanceModel

class Content(LanceModel):
    movie_id: int
    vector: Vector(128)
    genres: str
    title: str
    imdb_id: int

    @property
    def imdb_url(self) -> str:
        return f"https://www.imdb.com/title/tt{self.imdb_id}"

import pyarrow as pa
db = lancedb.connect("~/.lancedb")
table_name = "movielens_small"
table = db.create_table(table_name, schema=Content)
Nested schemas
Sometimes your data model may contain nested objects. For example, you may want to store the document string and the document soure name as a nested Document object:


class Document(BaseModel):
    content: str
    source: str
This can be used as the type of a LanceDB table column:


class NestedSchema(LanceModel):
    id: str
    vector: Vector(1536)
    document: Document

tbl = db.create_table("nested_table", schema=NestedSchema, mode="overwrite")
This creates a struct column called "document" that has two subfields called "content" and "source":


In [28]: tbl.schema
Out[28]:
id: string not null
vector: fixed_size_list<item: float>[1536] not null
    child 0, item: float
document: struct<content: string not null, source: string not null> not null
    child 0, content: string not null
    child 1, source: string not null
Validators
Note that neither Pydantic nor PyArrow automatically validates that input data is of the correct timezone, but this is easy to add as a custom field validator:


from datetime import datetime
from zoneinfo import ZoneInfo

from lancedb.pydantic import LanceModel
from pydantic import Field, field_validator, ValidationError, ValidationInfo

tzname = "America/New_York"
tz = ZoneInfo(tzname)

class TestModel(LanceModel):
    dt_with_tz: datetime = Field(json_schema_extra={"tz": tzname})

    @field_validator('dt_with_tz')
    @classmethod
    def tz_must_match(cls, dt: datetime) -> datetime:
        assert dt.tzinfo == tz
        return dt

ok = TestModel(dt_with_tz=datetime.now(tz))

try:
    TestModel(dt_with_tz=datetime.now(ZoneInfo("Asia/Shanghai")))
    assert 0 == 1, "this should raise ValidationError"
except ValidationError:
    print("A ValidationError was raised.")
    pass
When you run this code it should print "A ValidationError was raised."

Pydantic custom types
LanceDB does NOT yet support converting pydantic custom types. If this is something you need, please file a feature request on the LanceDB Github repo.

Using Iterators / Writing Large Datasets
It is recommended to use iterators to add large datasets in batches when creating your table in one go. This does not create multiple versions of your dataset unlike manually adding batches using table.add()

LanceDB additionally supports PyArrow's RecordBatch Iterators or other generators producing supported data types.

Here's an example using using RecordBatch iterator for creating tables.


import pyarrow as pa

def make_batches():
    for i in range(5):
        yield pa.RecordBatch.from_arrays(
            [
                pa.array([[3.1, 4.1, 5.1, 6.1], [5.9, 26.5, 4.7, 32.8]],
                        pa.list_(pa.float32(), 4)),
                pa.array(["foo", "bar"]),
                pa.array([10.0, 20.0]),
            ],
            ["vector", "item", "price"],
        )

schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 4)),
    pa.field("item", pa.utf8()),
    pa.field("price", pa.float32()),
])

db.create_table("batched_tale", make_batches(), schema=schema)
You can also use iterators of other types like Pandas DataFrame or Pylists directly in the above example.

Open existing tables

Python
Typescript1
If you forget the name of your table, you can always get a listing of all table names.


print(db.table_names())
Then, you can open any existing tables.


tbl = db.open_table("my_table")

Creating empty table
You can create an empty table for scenarios where you want to add data to the table later. An example would be when you want to collect data from a stream/external file and then add it to a table in batches.


Python
Typescript1
An empty table can be initialized via a PyArrow schema.


import lancedb
import pyarrow as pa

schema = pa.schema(
  [
      pa.field("vector", pa.list_(pa.float32(), 2)),
      pa.field("item", pa.string()),
      pa.field("price", pa.float32()),
  ])
tbl = db.create_table("empty_table_add", schema=schema)
Alternatively, you can also use Pydantic to specify the schema for the empty table. Note that we do not directly import pydantic but instead use lancedb.pydantic which is a subclass of pydantic.BaseModel that has been extended to support LanceDB specific types like Vector.


import lancedb
from lancedb.pydantic import LanceModel, vector

class Item(LanceModel):
    vector: Vector(2)
    item: str
    price: float

tbl = db.create_table("empty_table_add", schema=Item.to_arrow_schema())
Once the empty table has been created, you can add data to it via the various methods listed in the Adding to a table section.


Adding to a table
After a table has been created, you can always add more data to it usind the add method


Python
Typescript1
You can add any of the valid data structures accepted by LanceDB table, i.e, dict, list[dict], pd.DataFrame, or Iterator[pa.RecordBatch]. Below are some examples.

Add a Pandas DataFrame

df = pd.DataFrame({
    "vector": [[1.3, 1.4], [9.5, 56.2]], "item": ["banana", "apple"], "price": [5.0, 7.0]
})
tbl.add(df)
Add a Polars DataFrame

df = pl.DataFrame({
    "vector": [[1.3, 1.4], [9.5, 56.2]], "item": ["banana", "apple"], "price": [5.0, 7.0]
})
tbl.add(df)
Add an Iterator
You can also add a large dataset batch in one go using Iterator of any supported data types.


def make_batches():
    for i in range(5):
        yield [
                {"vector": [3.1, 4.1], "item": "peach", "price": 6.0},
                {"vector": [5.9, 26.5], "item": "pear", "price": 5.0}
            ]
tbl.add(make_batches())
Add a PyArrow table
If you have data coming in as a PyArrow table, you can add it directly to the LanceDB table.


pa_table = pa.Table.from_arrays(
        [
            pa.array([[9.1, 6.7], [9.9, 31.2]],
                    pa.list_(pa.float32(), 2)),
            pa.array(["mango", "orange"]),
            pa.array([7.0, 4.0]),
        ],
        ["vector", "item", "price"],
    )

tbl.add(pa_table)
Add a Pydantic Model
Assuming that a table has been created with the correct schema as shown above, you can add data items that are valid Pydantic models to the table.


pydantic_model_items = [
    Item(vector=[8.1, 4.7], item="pineapple", price=10.0),
    Item(vector=[6.9, 9.3], item="avocado", price=9.0)
]

tbl.add(pydantic_model_items)
Ingesting Pydantic models with LanceDB embedding API
When using LanceDB's embedding API, you can add Pydantic models directly to the table. LanceDB will automatically convert the vector field to a vector before adding it to the table. You need to specify the default value of vector feild as None to allow LanceDB to automatically vectorize the data.


import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

db = lancedb.connect("~/tmp")
embed_fcn = get_registry().get("huggingface").create(name="BAAI/bge-small-en-v1.5")

class Schema(LanceModel):
    text: str = embed_fcn.SourceField()
    vector: Vector(embed_fcn.ndims()) = embed_fcn.VectorField(default=None)

tbl = db.create_table("my_table", schema=Schema, mode="overwrite")
models = [Schema(text="hello"), Schema(text="world")]
tbl.add(models)

Deleting from a table
Use the delete() method on tables to delete rows from a table. To choose which rows to delete, provide a filter that matches on the metadata columns. This can delete any number of rows that match the filter.


Python
Typescript1

tbl.delete('item = "fizz"')
Deleting row with specific column value

import lancedb

data = [{"x": 1, "vector": [1, 2]},
        {"x": 2, "vector": [3, 4]},
        {"x": 3, "vector": [5, 6]}]
db = lancedb.connect("./.lancedb")
table = db.create_table("my_table", data)
table.to_pandas()
#   x      vector
# 0  1  [1.0, 2.0]
# 1  2  [3.0, 4.0]
# 2  3  [5.0, 6.0]

table.delete("x = 2")
table.to_pandas()
#   x      vector
# 0  1  [1.0, 2.0]
# 1  3  [5.0, 6.0]
Delete from a list of values

to_remove = [1, 5]
to_remove = ", ".join(str(v) for v in to_remove)

table.delete(f"x IN ({to_remove})")
table.to_pandas()
#   x      vector
# 0  3  [5.0, 6.0]

Updating a table
This can be used to update zero to all rows depending on how many rows match the where clause. The update queries follow the form of a SQL UPDATE statement. The where parameter is a SQL filter that matches on the metadata columns. The values or values_sql parameters are used to provide the new values for the columns.

Parameter	Type	Description
where	str	The SQL where clause to use when updating rows. For example, 'x = 2' or 'x IN (1, 2, 3)'. The filter must not be empty, or it will error.
values	dict	The values to update. The keys are the column names and the values are the values to set.
values_sql	dict	The values to update. The keys are the column names and the values are the SQL expressions to set. For example, {'x': 'x + 1'} will increment the value of the x column by 1.
SQL syntax

See SQL filters for more information on the supported SQL syntax.

Warning

Updating nested columns is not yet supported.


Python
Typescript1
API Reference: lancedb.table.Table.update


import lancedb
import pandas as pd

# Create a lancedb connection
db = lancedb.connect("./.lancedb")

# Create a table from a pandas DataFrame
data = pd.DataFrame({"x": [1, 2, 3], "vector": [[1, 2], [3, 4], [5, 6]]})
table = db.create_table("my_table", data)

# Update the table where x = 2
table.update(where="x = 2", values={"vector": [10, 10]})

# Get the updated table as a pandas DataFrame
df = table.to_pandas()

# Print the DataFrame
print(df)
Output


    x  vector
0  1  [1.0, 2.0]
1  3  [5.0, 6.0]
2  2  [10.0, 10.0]

Updating using a sql query
The values parameter is used to provide the new values for the columns as literal values. You can also use the values_sql / valuesSql parameter to provide SQL expressions for the new values. For example, you can use values_sql="x + 1" to increment the value of the x column by 1.


Python
Typescript1

# Update the table where x = 2
table.update(valuesSql={"x": "x + 1"})

print(table.to_pandas())
Output


    x  vector
0  2  [1.0, 2.0]
1  4  [5.0, 6.0]
2  3  [10.0, 10.0]

Note

When rows are updated, they are moved out of the index. The row will still show up in ANN queries, but the query will not be as fast as it would be if the row was in the index. If you update a large proportion of rows, consider rebuilding the index afterwards.

Drop a table
Use the drop_table() method on the database to remove a table.


Python
TypeScript

# Synchronous client
db.drop_table("my_table")
# Asynchronous client
await async_db.drop_table("my_table2")
This permanently removes the table and is not recoverable, unlike deleting rows. By default, if the table does not exist an exception is raised. To suppress this, you can pass in ignore_missing=True.


Consistency
In LanceDB OSS, users can set the read_consistency_interval parameter on connections to achieve different levels of read consistency. This parameter determines how frequently the database synchronizes with the underlying storage system to check for updates made by other processes. If another process updates a table, the database will not see the changes until the next synchronization.

There are three possible settings for read_consistency_interval:

Unset (default): The database does not check for updates to tables made by other processes. This provides the best query performance, but means that clients may not see the most up-to-date data. This setting is suitable for applications where the data does not change during the lifetime of the table reference.
Zero seconds (Strong consistency): The database checks for updates on every read. This provides the strongest consistency guarantees, ensuring that all clients see the latest committed data. However, it has the most overhead. This setting is suitable when consistency matters more than having high QPS.
Custom interval (Eventual consistency): The database checks for updates at a custom interval, such as every 5 seconds. This provides eventual consistency, allowing for some lag between write and read operations. Performance wise, this is a middle ground between strong consistency and no consistency check. This setting is suitable for applications where immediate consistency is not critical, but clients should see updated data eventually.
Consistency in LanceDB Cloud

This is only tune-able in LanceDB OSS. In LanceDB Cloud, readers are always eventually consistent.


Python
Typescript1
To set strong consistency, use timedelta(0):


from datetime import timedelta
db = lancedb.connect("./.lancedb",. read_consistency_interval=timedelta(0))
table = db.open_table("my_table")
For eventual consistency, use a custom timedelta:


from datetime import timedelta
db = lancedb.connect("./.lancedb", read_consistency_interval=timedelta(seconds=5))
table = db.open_table("my_table")
By default, a Table will never check for updates from other writers. To manually check for updates you can use checkout_latest:


db = lancedb.connect("./.lancedb")
table = db.open_table("my_table")

# (Other writes happen to my_table from another process)

# Check for updates
table.checkout_latest()