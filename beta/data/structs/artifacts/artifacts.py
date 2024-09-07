from .base import base

import pyarrow as pa


artifact = pa.schema(
    [
        ("name", pa.string()),
        ("type", pa.string()),
        ("content", pa.large_binary()),
        ("vectors", pa.list(pa.float64()))(
            "metadata", pa.map_(pa.string(), pa.string())
        ),
    ]
).append(base)
