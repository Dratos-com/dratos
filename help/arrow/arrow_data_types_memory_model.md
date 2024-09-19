Data Types and In-Memory Data Model
Apache Arrow defines columnar array data structures by composing type metadata with memory buffers, like the ones explained in the documentation on Memory and IO. These data structures are exposed in Python through a series of interrelated classes:

Type Metadata: Instances of pyarrow.DataType, which describe the type of an array and govern how its values are interpreted

Schemas: Instances of pyarrow.Schema, which describe a named collection of types. These can be thought of as the column types in a table-like object.

Arrays: Instances of pyarrow.Array, which are atomic, contiguous columnar data structures composed from Arrow Buffer objects

Record Batches: Instances of pyarrow.RecordBatch, which are a collection of Array objects with a particular Schema

Tables: Instances of pyarrow.Table, a logical table data structure in which each column consists of one or more pyarrow.Array objects of the same type.

We will examine these in the sections below in a series of examples.

Type Metadata
Apache Arrow defines language agnostic column-oriented data structures for array data. These include:

Fixed-length primitive types: numbers, booleans, date and times, fixed size binary, decimals, and other values that fit into a given number

Variable-length primitive types: binary, string

Nested types: list, map, struct, and union

Dictionary type: An encoded categorical type (more on this later)

Each data type in Arrow has a corresponding factory function for creating an instance of that type object in Python:

import pyarrow as pa

t1 = pa.int32()

t2 = pa.string()

t3 = pa.binary()

t4 = pa.binary(10)

t5 = pa.timestamp('ms')

t1
Out[7]: DataType(int32)

print(t1)
int32

print(t4)
fixed_size_binary[10]

print(t5)
timestamp[ms]
Note

Different data types might use a given physical storage. For example, int64, float64, and timestamp[ms] all occupy 64 bits per value.

These objects are metadata; they are used for describing the data in arrays, schemas, and record batches. In Python, they can be used in functions where the input data (e.g. Python objects) may be coerced to more than one Arrow type.

The Field type is a type plus a name and optional user-defined metadata:

f0 = pa.field('int32_field', t1)

f0
Out[12]: pyarrow.Field<int32_field: int32>

f0.name
Out[13]: 'int32_field'

f0.type
Out[14]: DataType(int32)
Arrow supports nested value types like list, map, struct, and union. When creating these, you must pass types or fields to indicate the data types of the types’ children. For example, we can define a list of int32 values with:

t6 = pa.list_(t1)

t6
Out[16]: ListType(list<item: int32>)
A struct is a collection of named fields:

fields = [
    pa.field('s0', t1),
    pa.field('s1', t2),
    pa.field('s2', t4),
    pa.field('s3', t6),
]


t7 = pa.struct(fields)

print(t7)
struct<s0: int32, s1: string, s2: fixed_size_binary[10], s3: list<item: int32>>
For convenience, you can pass (name, type) tuples directly instead of Field instances:

t8 = pa.struct([('s0', t1), ('s1', t2), ('s2', t4), ('s3', t6)])

print(t8)
struct<s0: int32, s1: string, s2: fixed_size_binary[10], s3: list<item: int32>>

t8 == t7
Out[22]: True
See Data Types API for a full listing of data type functions.

Schemas
The Schema type is similar to the struct array type; it defines the column names and types in a record batch or table data structure. The pyarrow.schema() factory function makes new Schema objects in Python:

my_schema = pa.schema([('field0', t1),
                       ('field1', t2),
                       ('field2', t4),
                       ('field3', t6)])


my_schema
Out[24]: 
field0: int32
field1: string
field2: fixed_size_binary[10]
field3: list<item: int32>
  child 0, item: int32
In some applications, you may not create schemas directly, only using the ones that are embedded in IPC messages.

Arrays
For each data type, there is an accompanying array data structure for holding memory buffers that define a single contiguous chunk of columnar array data. When you are using PyArrow, this data may come from IPC tools, though it can also be created from various types of Python sequences (lists, NumPy arrays, pandas data).

A simple way to create arrays is with pyarrow.array, which is similar to the numpy.array function. By default PyArrow will infer the data type for you:

arr = pa.array([1, 2, None, 3])

arr
Out[26]: 
<pyarrow.lib.Int64Array object at 0x7feac373d120>
[
  1,
  2,
  null,
  3
]
But you may also pass a specific data type to override type inference:

pa.array([1, 2], type=pa.uint16())
Out[27]: 
<pyarrow.lib.UInt16Array object at 0x7feac373d300>
[
  1,
  2
]
The array’s type attribute is the corresponding piece of type metadata:

arr.type
Out[28]: DataType(int64)
Each in-memory array has a known length and null count (which will be 0 if there are no null values):

len(arr)
Out[29]: 4

arr.null_count
Out[30]: 1
Scalar values can be selected with normal indexing. pyarrow.array converts None values to Arrow nulls; we return the special pyarrow.NA value for nulls:

arr[0]
Out[31]: <pyarrow.Int64Scalar: 1>

arr[2]
Out[32]: <pyarrow.Int64Scalar: None>
Arrow data is immutable, so values can be selected but not assigned.

Arrays can be sliced without copying:

arr[1:3]
Out[33]: 
<pyarrow.lib.Int64Array object at 0x7feac373dcc0>
[
  2,
  null
]
None values and NAN handling
As mentioned in the above section, the Python object None is always converted to an Arrow null element on the conversion to pyarrow.Array. For the float NaN value which is either represented by the Python object float('nan') or numpy.nan we normally convert it to a valid float value during the conversion. If an integer input is supplied to pyarrow.array that contains np.nan, ValueError is raised.

To handle better compatibility with Pandas, we support interpreting NaN values as null elements. This is enabled automatically on all from_pandas function and can be enabled on the other conversion functions by passing from_pandas=True as a function parameter.

List arrays
pyarrow.array is able to infer the type of simple nested data structures like lists:

nested_arr = pa.array([[], None, [1, 2], [None, 1]])

print(nested_arr.type)
list<item: int64>
ListView arrays
pyarrow.array can create an alternate list type called ListView:

nested_arr = pa.array([[], None, [1, 2], [None, 1]], type=pa.list_view(pa.int64()))

print(nested_arr.type)
list_view<item: int64>
ListView arrays have a different set of buffers than List arrays. The ListView array has both an offsets and sizes buffer, while a List array only has an offsets buffer. This allows for ListView arrays to specify out-of-order offsets:

values = [1, 2, 3, 4, 5, 6]

offsets = [4, 2, 0]

sizes = [2, 2, 2]

arr = pa.ListViewArray.from_arrays(offsets, sizes, values)

arr
Out[42]: 
<pyarrow.lib.ListViewArray object at 0x7feac373ec80>
[
  [
    5,
    6
  ],
  [
    3,
    4
  ],
  [
    1,
    2
  ]
]
See the format specification for more details on ListView Layout.

Struct arrays
pyarrow.array is able to infer the schema of a struct type from arrays of dictionaries:

pa.array([{'x': 1, 'y': True}, {'z': 3.4, 'x': 4}])
Out[43]: 
<pyarrow.lib.StructArray object at 0x7feac373f160>
-- is_valid: all not null
-- child 0 type: int64
  [
    1,
    4
  ]
-- child 1 type: bool
  [
    true,
    null
  ]
-- child 2 type: double
  [
    null,
    3.4
  ]
Struct arrays can be initialized from a sequence of Python dicts or tuples. For tuples, you must explicitly pass the type:

ty = pa.struct([('x', pa.int8()),
                ('y', pa.bool_())])


pa.array([{'x': 1, 'y': True}, {'x': 2, 'y': False}], type=ty)
Out[45]: 
<pyarrow.lib.StructArray object at 0x7feac373f580>
-- is_valid: all not null
-- child 0 type: int8
  [
    1,
    2
  ]
-- child 1 type: bool
  [
    true,
    false
  ]

pa.array([(3, True), (4, False)], type=ty)
Out[46]: 
<pyarrow.lib.StructArray object at 0x7feac373fac0>
-- is_valid: all not null
-- child 0 type: int8
  [
    3,
    4
  ]
-- child 1 type: bool
  [
    true,
    false
  ]
When initializing a struct array, nulls are allowed both at the struct level and at the individual field level. If initializing from a sequence of Python dicts, a missing dict key is handled as a null value:

pa.array([{'x': 1}, None, {'y': None}], type=ty)
Out[47]: 
<pyarrow.lib.StructArray object at 0x7feac373f2e0>
-- is_valid:
  [
    true,
    false,
    true
  ]
-- child 0 type: int8
  [
    1,
    0,
    null
  ]
-- child 1 type: bool
  [
    null,
    false,
    null
  ]
You can also construct a struct array from existing arrays for each of the struct’s components. In this case, data storage will be shared with the individual arrays, and no copy is involved:

xs = pa.array([5, 6, 7], type=pa.int16())

ys = pa.array([False, True, True])

arr = pa.StructArray.from_arrays((xs, ys), names=('x', 'y'))

arr.type
Out[51]: StructType(struct<x: int16, y: bool>)

arr
Out[52]: 
<pyarrow.lib.StructArray object at 0x7feac377c340>
-- is_valid: all not null
-- child 0 type: int16
  [
    5,
    6,
    7
  ]
-- child 1 type: bool
  [
    false,
    true,
    true
  ]
Map arrays
Map arrays can be constructed from lists of lists of tuples (key-item pairs), but only if the type is explicitly passed into array():

data = [[('x', 1), ('y', 0)], [('a', 2), ('b', 45)]]

ty = pa.map_(pa.string(), pa.int64())

pa.array(data, type=ty)
Out[55]: 
<pyarrow.lib.MapArray object at 0x7feac377c8e0>
[
  keys:
  [
    "x",
    "y"
  ]
  values:
  [
    1,
    0
  ],
  keys:
  [
    "a",
    "b"
  ]
  values:
  [
    2,
    45
  ]
]
MapArrays can also be constructed from offset, key, and item arrays. Offsets represent the starting position of each map. Note that the MapArray.keys and MapArray.items properties give the flattened keys and items. To keep the keys and items associated to their row, use the ListArray.from_arrays() constructor with the MapArray.offsets property.

arr = pa.MapArray.from_arrays([0, 2, 3], ['x', 'y', 'z'], [4, 5, 6])

arr.keys
Out[57]: 
<pyarrow.lib.StringArray object at 0x7feac377cb80>
[
  "x",
  "y",
  "z"
]

arr.items
Out[58]: 
<pyarrow.lib.Int64Array object at 0x7feac377cca0>
[
  4,
  5,
  6
]

pa.ListArray.from_arrays(arr.offsets, arr.keys)
Out[59]: 
<pyarrow.lib.ListArray object at 0x7feac377ce80>
[
  [
    "x",
    "y"
  ],
  [
    "z"
  ]
]

pa.ListArray.from_arrays(arr.offsets, arr.items)
Out[60]: 
<pyarrow.lib.ListArray object at 0x7feac377ce20>
[
  [
    4,
    5
  ],
  [
    6
  ]
]
Union arrays
The union type represents a nested array type where each value can be one (and only one) of a set of possible types. There are two possible storage types for union arrays: sparse and dense.

In a sparse union array, each of the child arrays has the same length as the resulting union array. They are adjuncted with a int8 “types” array that tells, for each value, from which child array it must be selected:

xs = pa.array([5, 6, 7])

ys = pa.array([False, False, True])

types = pa.array([0, 1, 1], type=pa.int8())

union_arr = pa.UnionArray.from_sparse(types, [xs, ys])

union_arr.type
Out[65]: SparseUnionType(sparse_union<0: int64=0, 1: bool=1>)

union_arr
Out[66]: 
<pyarrow.lib.UnionArray object at 0x7feac377d660>
-- is_valid: all not null
-- type_ids:   [
    0,
    1,
    1
  ]
-- child 0 type: int64
  [
    5,
    6,
    7
  ]
-- child 1 type: bool
  [
    false,
    false,
    true
  ]
In a dense union array, you also pass, in addition to the int8 “types” array, a int32 “offsets” array that tells, for each value, at each offset in the selected child array it can be found:

xs = pa.array([5, 6, 7])

ys = pa.array([False, True])

types = pa.array([0, 1, 1, 0, 0], type=pa.int8())

offsets = pa.array([0, 0, 1, 1, 2], type=pa.int32())

union_arr = pa.UnionArray.from_dense(types, offsets, [xs, ys])

union_arr.type
Out[72]: DenseUnionType(dense_union<0: int64=0, 1: bool=1>)

union_arr
Out[73]: 
<pyarrow.lib.UnionArray object at 0x7feac377d4e0>
-- is_valid: all not null
-- type_ids:   [
    0,
    1,
    1,
    0,
    0
  ]
-- value_offsets:   [
    0,
    0,
    1,
    1,
    2
  ]
-- child 0 type: int64
  [
    5,
    6,
    7
  ]
-- child 1 type: bool
  [
    false,
    true
  ]
Dictionary Arrays
The Dictionary type in PyArrow is a special array type that is similar to a factor in R or a pandas.Categorical. It enables one or more record batches in a file or stream to transmit integer indices referencing a shared dictionary containing the distinct values in the logical array. This is particularly often used with strings to save memory and improve performance.

The way that dictionaries are handled in the Apache Arrow format and the way they appear in C++ and Python is slightly different. We define a special DictionaryArray type with a corresponding dictionary type. Let’s consider an example:

indices = pa.array([0, 1, 0, 1, 2, 0, None, 2])

dictionary = pa.array(['foo', 'bar', 'baz'])

dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)

dict_array
Out[77]: 
<pyarrow.lib.DictionaryArray object at 0x7feac3767920>

-- dictionary:
  [
    "foo",
    "bar",
    "baz"
  ]
-- indices:
  [
    0,
    1,
    0,
    1,
    2,
    0,
    null,
    2
  ]
Here we have:

print(dict_array.type)
dictionary<values=string, indices=int64, ordered=0>

dict_array.indices
Out[79]: 
<pyarrow.lib.Int64Array object at 0x7feac373f5e0>
[
  0,
  1,
  0,
  1,
  2,
  0,
  null,
  2
]

dict_array.dictionary
Out[80]: 
<pyarrow.lib.StringArray object at 0x7feac373f820>
[
  "foo",
  "bar",
  "baz"
]
When using DictionaryArray with pandas, the analogue is pandas.Categorical (more on this later):

dict_array.to_pandas()
Out[81]: 
0    foo
1    bar
2    foo
3    bar
4    baz
5    foo
6    NaN
7    baz
dtype: category
Categories (3, object): ['foo', 'bar', 'baz']
Record Batches
A Record Batch in Apache Arrow is a collection of equal-length array instances. Let’s consider a collection of arrays:

data = [
    pa.array([1, 2, 3, 4]),
    pa.array(['foo', 'bar', 'baz', None]),
    pa.array([True, None, False, True])
]

A record batch can be created from this list of arrays using RecordBatch.from_arrays:

batch = pa.RecordBatch.from_arrays(data, ['f0', 'f1', 'f2'])

batch.num_columns
Out[84]: 3

batch.num_rows
Out[85]: 4

batch.schema
Out[86]: 
f0: int64
f1: string
f2: bool

batch[1]
Out[87]: 
<pyarrow.lib.StringArray object at 0x7feac377e200>
[
  "foo",
  "bar",
  "baz",
  null
]
A record batch can be sliced without copying memory like an array:

batch2 = batch.slice(1, 3)

batch2[1]
Out[89]: 
<pyarrow.lib.StringArray object at 0x7feac377ed40>
[
  "bar",
  "baz",
  null
]
Tables
The PyArrow Table type is not part of the Apache Arrow specification, but is rather a tool to help with wrangling multiple record batches and array pieces as a single logical dataset. As a relevant example, we may receive multiple small record batches in a socket stream, then need to concatenate them into contiguous memory for use in NumPy or pandas. The Table object makes this efficient without requiring additional memory copying.

Considering the record batch we created above, we can create a Table containing one or more copies of the batch using Table.from_batches:

batches = [batch] * 5

table = pa.Table.from_batches(batches)

table
Out[92]: 
pyarrow.Table
f0: int64
f1: string
f2: bool
----
f0: [[1,2,3,4],[1,2,3,4],...,[1,2,3,4],[1,2,3,4]]
f1: [["foo","bar","baz",null],["foo","bar","baz",null],...,["foo","bar","baz",null],["foo","bar","baz",null]]
f2: [[true,null,false,true],[true,null,false,true],...,[true,null,false,true],[true,null,false,true]]

table.num_rows
Out[93]: 20
The table’s columns are instances of ChunkedArray, which is a container for one or more arrays of the same type.

c = table[0]

c
Out[95]: 
<pyarrow.lib.ChunkedArray object at 0x7feac37abe20>
[
  [
    1,
    2,
    3,
    4
  ],
  [
    1,
    2,
    3,
    4
  ],
...,
  [
    1,
    2,
    3,
    4
  ],
  [
    1,
    2,
    3,
    4
  ]
]

c.num_chunks
Out[96]: 5

c.chunk(0)
Out[97]: 
<pyarrow.lib.Int64Array object at 0x7feac377f580>
[
  1,
  2,
  3,
  4
]
As you’ll see in the pandas section, we can convert these objects to contiguous NumPy arrays for use in pandas:

c.to_pandas()
Out[98]: 
0     1
1     2
2     3
3     4
4     1
5     2
6     3
7     4
8     1
9     2
10    3
11    4
12    1
13    2
14    3
15    4
16    1
17    2
18    3
19    4
Name: f0, dtype: int64
Multiple tables can also be concatenated together to form a single table using pyarrow.concat_tables, if the schemas are equal:

tables = [table] * 2

table_all = pa.concat_tables(tables)

table_all.num_rows
Out[101]: 40

c = table_all[0]

c.num_chunks
Out[103]: 10
This is similar to Table.from_batches, but uses tables as input instead of record batches. Record batches can be made into tables, but not the other way around, so if your data is already in table form, then use pyarrow.concat_tables.

Custom Schema and Field Metadata
Arrow supports both schema-level and field-level custom key-value metadata allowing for systems to insert their own application defined metadata to customize behavior.

Custom metadata can be accessed at Schema.metadata for the schema-level and Field.metadata for the field-level.

Note that this metadata is preserved in Streaming, Serialization, and IPC processes.

To customize the schema metadata of an existing table you can use Table.replace_schema_metadata():

table.schema.metadata # empty

table = table.replace_schema_metadata({"f0": "First dose"})

table.schema.metadata
Out[106]: {b'f0': b'First dose'}
To customize the metadata of the field from the table schema you can use Field.with_metadata():

field_f1 = table.schema.field("f1")

field_f1.metadata # empty

field_f1 = field_f1.with_metadata({"f1": "Second dose"})

field_f1.metadata
Out[110]: {b'f1': b'Second dose'}
Both options create a shallow copy of the data and do not in fact change the Schema which is immutable. To change the metadata in the schema of the table we created a new object when calling Table.replace_schema_metadata().

To change the metadata of the field in the schema we would need to define a new schema and cast the data to this schema:

my_schema2 = pa.schema([
   pa.field('f0', pa.int64(), metadata={"name": "First dose"}),
   pa.field('f1', pa.string(), metadata={"name": "Second dose"}),
   pa.field('f2', pa.bool_())],
   metadata={"f2": "booster"})


t2 = table.cast(my_schema2)

t2.schema.field("f0").metadata
Out[113]: {b'name': b'First dose'}

t2.schema.field("f1").metadata
Out[114]: {b'name': b'Second dose'}

t2.schema.metadata
Out[115]: {b'f2': b'booster'}
Metadata key and value pairs are std::string objects in the C++ implementation and so they are bytes objects (b'...') in Python.

Record Batch Readers
Many functions in PyArrow either return or take as an argument a RecordBatchReader. It can be used like any iterable of record batches, but also provides their common schema without having to get any of the batches.:

schema = pa.schema([('x', pa.int64())])
def iter_record_batches():
   for i in range(2):
      yield pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], schema=schema)
reader = pa.RecordBatchReader.from_batches(schema, iter_record_batches())
print(reader.schema)
pyarrow.Schema
x: int64
for batch in reader:
   print(batch)
pyarrow.RecordBatch
x: int64
pyarrow.RecordBatch
x: int64
It can also be sent between languages using the C stream interface.

Conversion of RecordBatch to Tensor
Each array of the RecordBatch has it’s own contiguous memory that is not necessarily adjacent to other arrays. A different memory structure that is used in machine learning libraries is a two dimensional array (also called a 2-dim tensor or a matrix) which takes only one contiguous block of memory.

For this reason there is a function pyarrow.RecordBatch.to_tensor() available to efficiently convert tabular columnar data into a tensor.

Data types supported in this conversion are unsigned, signed integer and float types. Currently only column-major conversion is supported.

 import pyarrow as pa
 arr1 = [1, 2, 3, 4, 5]
 arr2 = [10, 20, 30, 40, 50]
 batch = pa.RecordBatch.from_arrays(
     [
         pa.array(arr1, type=pa.uint16()),
         pa.array(arr2, type=pa.int16()),
     ], ["a", "b"]
 )
 batch.to_tensor()
<pyarrow.Tensor>
type: int32
shape: (9, 2)
strides: (4, 36)
 batch.to_tensor().to_numpy()
array([[ 1, 10],
      [ 2, 20],
      [ 3, 30],
      [ 4, 40],
      [ 5, 50]], dtype=int32)
With null_to_nan set to True one can also convert data with nulls. They will be converted to NaN:

import pyarrow as pa
batch = pa.record_batch(
    [
        pa.array([1, 2, 3, 4, None], type=pa.int32()),
        pa.array([10, 20, 30, 40, None], type=pa.float32()),
    ], names = ["a", "b"]
)
batch.to_tensor(null_to_nan=True).to_numpy()
array([[ 1., 10.],
      [ 2., 20.],
      [ 3., 30.],
      [ 4., 40.],
      [nan, nan]])