from .base import base
import pyarrow as pa

model_schema = pa.schema(
    [
        ("name", pa.string()),
        ("description", pa.string()),
        ("parameters", pa.map_(pa.string(), pa.string())),
        ("performance_metrics", pa.map_(pa.string(), pa.float64())),
    ]
).append(base)

llm = pa.schema(
    [
        ("model_name", pa.string()),
        ("provider", pa.string()),
        ("version", pa.string()),
        ("parameters", pa.map_(pa.string(), pa.string())),
        ("capabilities", pa.list_(pa.string())),
        ("fine_tuned", pa.bool_()),
        ("training_data", pa.string()),
        ("performance_metrics", pa.map_(pa.string(), pa.float64())),
    ]
).append(model_schema)


vllm_settings = pa.schema(
    [
        ("max_tokens", pa.int32()),
        ("temperature", pa.float32()),
        ("top_p", pa.float32()),
        ("top_k", pa.int32()),
        ("presence_penalty", pa.float32()),
        ("frequency_penalty", pa.float32()),
        ("repetition_penalty", pa.float32()),
        ("stop", pa.list_(pa.string())),
        ("best_of", pa.int32()),
        ("use_beam_search", pa.bool_()),
        ("length_penalty", pa.float32()),
        ("early_stopping", pa.bool_()),
        ("seed", pa.int64()),
        ("logprobs", pa.int32()),
        ("echo", pa.bool_()),
    ]
).append(base)
