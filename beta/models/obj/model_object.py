from __future__ import annotations

import io
from typing import Optional, Dict, Any, ClassVar, TypeVar, Iterator
import pyarrow as pa
from pydantic import Field
from outlines.models.vllm import VLLM
from dataclasses import dataclass
import daft
from beta.data.obj.base.data_object import DataObject  
import torch
import numpy as np


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.backends.tpu.is_available():
        return torch.device("tpu")
    else:
        return torch.device("cpu")

class Model:
    __tablename__: ClassVar[str] = "models"

    model_type: str = Field(..., description="Type of ML model")
    model_version: str = Field(..., description="Version of ML model")
    model_config: Dict[str, Any] = Field(..., description="Configuration of ML model")
    model_weights: Optional[bytes] = Field(
        default=None, description="Weights of ML model"
    )
    model_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata of ML model"
    )

    def __init__(self, config: daft.DataFrame = None, **data: Any, ) -> None:
        super().__init__(**data)

    def save(self, namespace: str, table_name: str) -> None:
        self.manager.save_data_objects([self], namespace, table_name)
    
    @daft.udf(return_dtype=daft.DataType.from_arrow_type(pa.float64()))
    def predict(self, data: daft.DataFrame) -> daft.DataFrame:
        # Load the model weights if they're not None
        if self.model_weights is not None:
            # Deserialize the model weights
            model = torch.load(io.BytesIO(self.model_weights), map_location=get_device())
            model.eval()  # Set the model to evaluation mode
        else:
            raise ValueError("Model weights are not available")

        # Convert Daft DataFrame to PyTorch tensor
        input_tensor = torch.tensor(data["x"].to_numpy(), dtype=torch.float32).to(get_device())

        # Perform prediction
        with torch.no_grad():
            output = model(input_tensor)

        # Convert the output back to a Daft DataFrame
        return daft.from_pydict({"prediction": output.cpu().numpy().flatten().tolist()})

    @classmethod
    def load(cls, namespace: str, table_name: str):
        data_objects = cls.manager.get_data_objects(namespace, table_name)
        if len(data_objects) == 0:
            raise ValueError(
                f"No ML model found in namespace {namespace} and table {table_name}"
            )
        return data_objects[0]



