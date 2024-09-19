Memory and IO Interfaces
This section will introduce you to the major concepts in PyArrow’s memory management and IO systems:

Buffers

Memory pools

File-like and stream-like objects

Referencing and Allocating Memory
pyarrow.Buffer
The Buffer object wraps the C++ arrow::Buffer type which is the primary tool for memory management in Apache Arrow in C++. It permits higher-level array classes to safely interact with memory which they may or may not own. arrow::Buffer can be zero-copy sliced to permit Buffers to cheaply reference other Buffers, while preserving memory lifetime and clean parent-child relationships.

There are many implementations of arrow::Buffer, but they all provide a standard interface: a data pointer and length. This is similar to Python’s built-in buffer protocol and memoryview objects.

A Buffer can be created from any Python object implementing the buffer protocol by calling the py_buffer() function. Let’s consider a bytes object:

import pyarrow as pa

data = b'abcdefghijklmnopqrstuvwxyz'

buf = pa.py_buffer(data)

buf
Out[4]: <pyarrow.Buffer address=0x7fea802f5d10 size=26 is_cpu=True is_mutable=False>

buf.size
Out[5]: 26
Creating a Buffer in this way does not allocate any memory; it is a zero-copy view on the memory exported from the data bytes object.

External memory, under the form of a raw pointer and size, can also be referenced using the foreign_buffer() function.

Buffers can be used in circumstances where a Python buffer or memoryview is required, and such conversions are zero-copy:

memoryview(buf)
Out[6]: <memory at 0x7fea81acdf00>
The Buffer’s to_pybytes() method converts the Buffer’s data to a Python bytestring (thus making a copy of the data):

buf.to_pybytes()
Out[7]: b'abcdefghijklmnopqrstuvwxyz'
Memory Pools
All memory allocations and deallocations (like malloc and free in C) are tracked in an instance of MemoryPool. This means that we can then precisely track amount of memory that has been allocated:

pa.total_allocated_bytes()
Out[8]: 56320
Let’s allocate a resizable Buffer from the default pool:

buf = pa.allocate_buffer(1024, resizable=True)

pa.total_allocated_bytes()
Out[10]: 57344

buf.resize(2048)

pa.total_allocated_bytes()
Out[12]: 58368
The default allocator requests memory in a minimum increment of 64 bytes. If the buffer is garbage-collected, all of the memory is freed:

buf = None

pa.total_allocated_bytes()
Out[14]: 56320
Besides the default built-in memory pool, there may be additional memory pools to choose (such as mimalloc) from depending on how Arrow was built. One can get the backend name for a memory pool:

pa.default_memory_pool().backend_name
'jemalloc'
See also

API documentation for memory pools.

See also

On-GPU buffers using Arrow’s optional CUDA integration.

Input and Output
The Arrow C++ libraries have several abstract interfaces for different kinds of IO objects:

Read-only streams

Read-only files supporting random access

Write-only streams

Write-only files supporting random access

File supporting reads, writes, and random access

In the interest of making these objects behave more like Python’s built-in file objects, we have defined a NativeFile base class which implements the same API as regular Python file objects.

NativeFile has some important features which make it preferable to using Python files with PyArrow where possible:

Other Arrow classes can access the internal C++ IO objects natively, and do not need to acquire the Python GIL

Native C++ IO may be able to do zero-copy IO, such as with memory maps

There are several kinds of NativeFile options available:

OSFile, a native file that uses your operating system’s file descriptors

MemoryMappedFile, for reading (zero-copy) and writing with memory maps

BufferReader, for reading Buffer objects as a file

BufferOutputStream, for writing data in-memory, producing a Buffer at the end

FixedSizeBufferWriter, for writing data into an already allocated Buffer

HdfsFile, for reading and writing data to the Hadoop Filesystem

PythonFile, for interfacing with Python file objects in C++

CompressedInputStream and CompressedOutputStream, for on-the-fly compression or decompression to/from another stream

There are also high-level APIs to make instantiating common kinds of streams easier.

High-Level API
Input Streams
The input_stream() function allows creating a readable NativeFile from various kinds of sources.

If passed a Buffer or a memoryview object, a BufferReader will be returned:

buf = memoryview(b"some data")

stream = pa.input_stream(buf)

stream.read(4)
Out[17]: b'some'
If passed a string or file path, it will open the given file on disk for reading, creating a OSFile. Optionally, the file can be compressed: if its filename ends with a recognized extension such as .gz, its contents will automatically be decompressed on reading.

import gzip

with gzip.open('example.gz', 'wb') as f:
    f.write(b'some data\n' * 3)


stream = pa.input_stream('example.gz')

stream.read()
Out[21]: b'some data\nsome data\nsome data\n'
If passed a Python file object, it will wrapped in a PythonFile such that the Arrow C++ libraries can read data from it (at the expense of a slight overhead).

Output Streams
output_stream() is the equivalent function for output streams and allows creating a writable NativeFile. It has the same features as explained above for input_stream(), such as being able to write to buffers or do on-the-fly compression.

with pa.output_stream('example1.dat') as stream:
    stream.write(b'some data')


f = open('example1.dat', 'rb')

f.read()
Out[24]: b'some data'
On-Disk and Memory Mapped Files
PyArrow includes two ways to interact with data on disk: standard operating system-level file APIs, and memory-mapped files. In regular Python we can write:

with open('example2.dat', 'wb') as f:
    f.write(b'some example data')

Using pyarrow’s OSFile class, you can write:

with pa.OSFile('example3.dat', 'wb') as f:
    f.write(b'some example data')

For reading files, you can use OSFile or MemoryMappedFile. The difference between these is that OSFile allocates new memory on each read, like Python file objects. In reads from memory maps, the library constructs a buffer referencing the mapped memory without any memory allocation or copying:

file_obj = pa.OSFile('example2.dat')

mmap = pa.memory_map('example3.dat')

file_obj.read(4)
Out[29]: b'some'

mmap.read(4)
Out[30]: b'some'
The read method implements the standard Python file read API. To read into Arrow Buffer objects, use read_buffer:

mmap.seek(0)
Out[31]: 0

buf = mmap.read_buffer(4)

print(buf)
<pyarrow.Buffer address=0x7feb22c88000 size=4 is_cpu=True is_mutable=False>

buf.to_pybytes()
Out[34]: b'some'
Many tools in PyArrow, particular the Apache Parquet interface and the file and stream messaging tools, are more efficient when used with these NativeFile types than with normal Python file objects.

In-Memory Reading and Writing
To assist with serialization and deserialization of in-memory data, we have file interfaces that can read and write to Arrow Buffers.

writer = pa.BufferOutputStream()

writer.write(b'hello, friends')
Out[36]: 14

buf = writer.getvalue()

buf
Out[38]: <pyarrow.Buffer address=0x7feb19613000 size=14 is_cpu=True is_mutable=True>

buf.size
Out[39]: 14

reader = pa.BufferReader(buf)

reader.seek(7)
Out[41]: 7

reader.read(7)
Out[42]: b'friends'
These have similar semantics to Python’s built-in io.BytesIO.