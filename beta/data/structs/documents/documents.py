from ...schema.base import base
from .artifacts import artifact
import pyarrow as pa

node = pa.schema(
    [
        pa.field("content", pa.string()),
        pa.field("properties", pa.map_(pa.string(), pa.string())),
        pa.field("incoming_edges", pa.list_(pa.string())),
        pa.field("outgoing_edges", pa.list_(pa.string())),
    ]
).append(base)

# Index schema
index = pa.schema(
    [
        ("value", pa.string()),
        ("type", pa.string()),
        ("fields", pa.structArray(pa.string())),
    ]
).append(base)

# Document schema
document = pa.schema(
    [
        ("title", pa.string()),
        ("content", pa.long_string()),
        ("nodes", pa.list_(node)),
        ("indices", pa.list_(index)),
        ("embeddings", pa.list(pa.float32())),
    ]
).append(base)
