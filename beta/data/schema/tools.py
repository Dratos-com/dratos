from .base import base
import pyarrow as pa

tool_schema = pa.schema(
    [
        ("name", pa.string()),
        ("description", pa.string()),
        ("parameters", pa.map_(pa.string(), pa.string())),
        ("output_type", pa.string()),
    ]
).append(base_schema)
