from __future__ import annotations

from typing import Optional, Dict, Any, ClassVar, Type, TypeVar, List, Union
import pyarrow as pa
from datetime import datetime
import ulid
import json
from pydantic import Field
from outlines.models import BaseModel as OutlinesBaseModel
from dataclasses import dataclass
import daft
import typing
from beta.data.obj.data_object import DataObject
from beta.data.obj.data_object_accessor import DataObjectAccessor
from beta.data.obj.base. import DataObjectManager    
import torch
import numpy as np

T = TypeVar("T", bound="DataObject")

@daft.udf(input_type=daft.DataType.from_arrow_type(pa.struct({"x": pa.float64()})), output_type=daft.DataType.from_arrow_type(pa.float64()))
def predict(x: float) -> float:
    return x * 2

class ModelObject(DataObject):
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
    
    @daft.udf(io)
    def predict(self, data: daft.DataFrame) -> daft.DataFrame:
        

    @classmethod
    def load(cls, namespace: str, table_name: str) -> BaseMLModel:
        data_objects = cls.manager.get_data_objects(namespace, table_name)
        if len(data_objects) == 0:
            raise ValueError(
                f"No ML model found in namespace {namespace} and table {table_name}"
            )
        return data_objects[0]



def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.backends.tpu.is_available():
        return torch.device("tpu")
    else:
        return torch.device("cpu")

class StateSpaceModel(ModelObject):
    def __init__(self, A, B, C, D):
        self.A = A  # State transition matrix
        self.B = B  # Control matrix
        self.C = C  # Observation matrix
        self.D = D  # Feedforward matrix
        self.time_series_data = []

    def record_time_series(self, data: daft.DataFrame, output: daft.DataFrame) -> None:
        """
        Record the time series data for the model in a production-ready manner.

        Args:
            data: A Daft DataFrame containing the input data.
            output: A Daft DataFrame containing the predicted output.
        """
        # Use Deltacat for production-ready time series recording
        self.time_series_data.append(
            {
                "t": data["t"].to_numpy()[0],
                "u": data["u"].to_numpy()[0],
                "y_pred": output["y_pred"].to_numpy()[0],
            }
        )
    

    def controllability_matrix(self):
        """Calculate and return the controllability matrix."""
        import daft
        import pyarrow as pa
        import torch
        try:
            device = get_device()
            A = torch.tensor(self.A).to(device)
            B = torch.tensor(self.B).to(device)
            return daft.DataFrame.from_arrow(pa.table(
                [
                    pa.array(
                        [
                            torch.linalg.matrix_power(A, i) @ B
                            for i in range(self.A.shape[0])
                        ]
                    )
                ],
                schema=pa.schema(
                    [
                        pa.field(
                            f"controllability_matrix_{i}",
                            pa.list_(pa.float64()),
                        )
                        for i in range(self.A.shape[0])
                    ]
                ),
            ))
        except Exception as e:
            raise ValueError(f"Error calculating controllability matrix: {e}")
        pass

    def observability_matrix(self):
        """Calculate and return the observability matrix."""
        Mirror controllability_matrix
        import daft
        import pyarrow as pa
        import torch
        try:
            device = get_device()
            A = torch.tensor(self.A).to(device)
            C = torch.tensor(self.C).to(device)
            return daft.DataFrame.from_arrow(pa.table(
                [
                    pa.array(
                        [
                            torch.linalg.matrix_power(A, i) @ C
                            for i in range(self.A.shape[0])
                        ]
                    )
                ],
                schema=pa.schema(
                    [
                        pa.field(
                            f"observability_matrix_{i}",
                            pa.list_(pa.float64()),
                        )
                        for i in range(self.A.shape[0])
                    ]
                ),
            ))
        except Exception as e:
            raise ValueError(f"Error calculating observability matrix: {e}")


class StreamingStateSpaceModel(ModelObject):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.time_series_data = []
    def predict(self, data: daft.DataFrame) -> daft.DataFrame:
        """
        Predict the output of the state-space model given input data.

        Args:
            data: A Daft DataFrame containing the input data.

        Returns:
            A Daft DataFrame containing the predicted output.
        """
        # Extract the input data from the DataFrame
        u = data["u"].to_numpy()
        # Initialize the state vector
        x = np.zeros((self.A.shape[0], 1))
        # Create an empty list to store the predicted output
        y_pred = []
        # Iterate over the input data
        for i in range(len(u)):
            # Update the state vector
            x = self.A @ x + self.B @ u[i]
            # Predict the output
            y = self.C @ x + self.D @ u[i]
            # Append the predicted output to the list
            y_pred.append(y)
        # Convert the predicted output to a Daft DataFrame
        y_pred = daft.DataFrame.from_arrow(pa.table({"y_pred": y_pred}))
        # Return the predicted output
        return y_pred

    def predict_stream(self, data: daft.DataFrame) -> daft.DataFrame:
        """
        Predict the output of the state-space model given input data in a streaming fashion.

        Args:
            data: A Daft DataFrame containing the input data.

        Returns:
            A Daft DataFrame containing the predicted output.
        """
        # Extract the input data from the DataFrame
        u = data["u"].to_numpy()
        # Initialize the state vector
        x = np.zeros((self.A.shape[0], 1))
        # Create an empty list to store the predicted output
        y_pred = []
        # Iterate over the input data
        for i in range(len(u)):
            # Update the state vector
            x = self.A @ x + self.B @ u[i]
            # Predict the output
            y = self.C @ x + self.D @ u[i]
            # Append the predicted output to the list
            y_pred.append(y)
            # Yield the predicted output
            yield daft.DataFrame.from_arrow(pa.table({"y_pred": [y]}))
        # Record the time series data
        self.time_series_data.append(
            {
                "t": data["t"].to_numpy()[0],
                "u": data["u"].to_numpy()[0],
                "y_pred": y,
            }
        )










class LinearRegressionModel(ModelObject):
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    def predict(self, data: daft.DataFrame) -> daft.DataFrame:
        """
        Predict the output of the linear regression model given input data.

        Args:
            data: A Daft DataFrame containing the input data.

        Returns:
            A Daft DataFrame containing the predicted output.
        """
        # Extract the input data from the DataFrame
        X = data["X"].to_numpy()
        # Use the model weights to predict the output
        y_pred = X @ self.model_weights
        # Convert the predicted output to a Daft DataFrame
        return daft.DataFrame.from_arrow(pa.table({"y_pred": y_pred}))
    
    