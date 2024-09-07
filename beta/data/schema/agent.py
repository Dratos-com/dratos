from .base import base
from .artifacts import artifact
from ..structs.documents.documents import document
from .models import llm
from .tools import tool

import pyarrow as pa


agent = pa.schema(
    [
        ("name", pa.string()),
        ("description", pa.string()),
        ("role", pa.string()),
        ("persona", pa.string()),
        ("capabilities", pa.list_(pa.string())),
        ("knowledge_base", pa.list_(document)),
        ("model", llm),
        ("tools", pa.list_(tool)),
        ("state", pa.map_(pa.string(), pa.string())),
    ]
).append(base)
