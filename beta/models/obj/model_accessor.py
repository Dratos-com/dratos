from typing import Type, List, Optional, Dict, Any
import numpy as np
import pyarrow as pa
import lance
from lance import LanceDataset
from beta.data.obj.base.data_object import DataObject
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelAccessor:
    """
    Accessor for model data, capable of handling numpy arrays and storing them in LanceDB.
    Includes MLflow integration for experiment tracking.
    """

    def __init__(
        self,
        data_object_class: Type[DataObject],
        dataset_uri: str,
        experiment_name: str,
    ):
        self.data_object_class = data_object_class
        self.dataset_uri = dataset_uri
        self._ensure_dataset_exists()
        self.experiment_name = experiment_name
        self._setup_mlflow()

    def _ensure_dataset_exists(self):
        """Ensure that the LanceDB dataset exists, create it if it doesn't."""
        try:
            self.dataset = LanceDataset(uri=self.dataset_uri)
            logger.info(f"Connected to existing dataset at {self.dataset_uri}")
        except lance.dataset.DatasetNotFoundException:
            self._create_dataset()

    def _create_dataset(self):
        """Create a new LanceDB dataset with the schema of the DataObject."""
        schema = self.data_object_class.get_arrow_schema()
        self.dataset = LanceDataset.create(self.dataset_uri, schema=schema)
        logger.info(f"Created new dataset at {self.dataset_uri}")

    def _setup_mlflow(self):
        """Set up MLflow experiment."""
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Set up MLflow experiment: {self.experiment_name}")

    def save_prediction(
        self,
        data_objects: List[DataObject],
        predictions: np.ndarray,
        model_params: Dict[str, Any] = None,
    ):
        """Save model predictions as numpy arrays to LanceDB and log to MLflow."""
        with mlflow.start_run(nested=True):
            arrow_table = self.data_object_class.to_arrow_table(data_objects)
            prediction_array = pa.array(predictions, type=pa.float64())
            combined_table = pa.Table.from_arrays(
                arrow_table.columns + [prediction_array],
                names=arrow_table.column_names + ["predictions"],
            )
            self.dataset.write(combined_table)

            # Log parameters
            if model_params:
                mlflow.log_params(model_params)

            # Log metrics
            mlflow.log_metric("num_predictions", len(predictions))
            mlflow.log_metric("mean_prediction", np.mean(predictions))
            mlflow.log_metric("std_prediction", np.std(predictions))

            logger.info(
                f"Saved {len(predictions)} predictions to dataset and logged to MLflow"
            )

    def get_data(self, filter_expr: Optional[pa.Expression] = None) -> List[DataObject]:
        """Retrieve data objects, optionally filtered by a PyArrow expression."""
        scanner = self.dataset.scanner(filter=filter_expr)
        arrow_table = scanner.to_table()
        logger.info(f"Retrieved {len(arrow_table)} rows from dataset")
        return self.data_object_class.from_arrow_table(arrow_table)

    def get_data_object_by_id(self, object_id: str) -> DataObject:
        """Retrieve a single data object by its ID."""
        filter_expr = pa.field("id") == object_id
        results = self.get_data(filter_expr=filter_expr)
        if not results:
            logger.warning(f"No object found with id: {object_id}")
            raise ValueError(f"No object found with id: {object_id}")
        logger.info(f"Retrieved object with id: {object_id}")
        return results[0]

    def write_data_objects(self, data_objects: List[DataObject]):
        """Write a list of data objects to the LanceDB dataset."""
        arrow_table = self.data_object_class.to_arrow_table(data_objects)
        self.dataset.write(arrow_table)
        logger.info(f"Wrote {len(data_objects)} objects to dataset")

    def update_data_object(self, data_object: DataObject):
        """Update a single data object in the LanceDB dataset."""
        arrow_table = self.data_object_class.to_arrow_table([data_object])
        self.dataset.merge_insert(arrow_table, on=["id"])
        logger.info(f"Updated object with id: {data_object.id}")

    def delete_data_object(self, object_id: str):
        """Delete a single data object from the LanceDB dataset."""
        self.dataset.delete(pa.field("id") == object_id)
        logger.info(f"Deleted object with id: {object_id}")

    def filter(self, filter_expr: pa.Expression) -> pa.Table:
        """Filter the data using a PyArrow expression."""
        result = self.dataset.scanner(filter=filter_expr).to_table()
        logger.info(f"Filtered data, resulting in {len(result)} rows")
        return result

    def select(self, *columns: str) -> pa.Table:
        """Select specific columns from the data."""
        result = self.dataset.scanner(columns=list(columns)).to_table()
        logger.info(f"Selected columns: {columns}, resulting in {len(result)} rows")
        return result

    def sort(self, by: List[str], ascending: List[bool] = None) -> pa.Table:
        """Sort the data by specified columns."""
        result = self.dataset.scanner().sort_by(by, ascending=ascending).to_table()
        logger.info(f"Sorted data by {by}, resulting in {len(result)} rows")
        return result

    def limit(self, n: int) -> pa.Table:
        """Limit the number of rows in the data."""
        result = self.dataset.scanner().limit(n).to_table()
        logger.info(f"Limited data to {n} rows")
        return result

    def create_index(self, column: str, index_type: str = "vector"):
        """Create an index on a specific column."""
        self.dataset.create_index(column, index_type=index_type)
        logger.info(f"Created {index_type} index on column: {column}")

    def nearest(self, column: str, query_vector: np.ndarray, k: int = 10) -> pa.Table:
        """Perform a nearest neighbor search on a vector column."""
        result = self.dataset.scanner().nearest(column, query_vector, k=k).to_table()
        logger.info(
            f"Performed nearest neighbor search on {column}, returned {len(result)} results"
        )
        return result

    def to_pandas(self) -> "pandas.DataFrame":
        """Convert the data to a pandas DataFrame."""
        result = self.dataset.to_table().to_pandas()
        logger.info(f"Converted dataset to pandas DataFrame with {len(result)} rows")
        return result

    def to_arrow(self) -> pa.Table:
        """Convert the data to a PyArrow Table."""
        result = self.dataset.to_table()
        logger.info(f"Converted dataset to PyArrow Table with {len(result)} rows")
        return result

    def optimize(self):
        """Optimize the dataset for better query performance."""
        self.dataset.optimize()
        logger.info("Optimized dataset for better query performance")

    def log_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """Log model performance metrics to MLflow."""
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.log_param("model_name", model_name)
        logger.info(f"Logged performance metrics for model: {model_name}")
