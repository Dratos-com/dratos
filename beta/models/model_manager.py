from typing import List, Dict, Any, Optional
import numpy as np
from beta.models.obj.model_accessor import ModelAccessor
from beta.models.types.base_ml_model import BaseMLModel
from beta.config.config import Config
import mlflow


class ModelManager:
    def __init__(self, dataset_uri: str, experiment_name: str):
        self.config = Config.get_instance()
        self.mlflow_client = self.config.get_mlflow()
        self.model_accessor = ModelAccessor(BaseMLModel, dataset_uri, experiment_name)
        self.mlflow_client.set_experiment(experiment_name)

    def save_model(self, model: BaseMLModel, model_params: Dict[str, Any]):
        """Save the model and log its parameters."""
        with self.mlflow_client.start_run():
            self.model_accessor.write_data_objects([model])
            self.mlflow_client.log_params(model_params)
            self.mlflow_client.log_artifact(model.model_weights, "model_weights")

    def load_model(self, model_id: str) -> BaseMLModel:
        """Load a model by its ID."""
        return self.model_accessor.get_data_object_by_id(model_id)

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics for the current run."""
        with self.mlflow_client.start_run(nested=True):
            self.mlflow_client.log_metrics(metrics)

    def log_predictions(self, data_objects: List[BaseMLModel], predictions: np.ndarray):
        """Log predictions and associated data objects."""
        self.model_accessor.save_prediction(data_objects, predictions)

    def get_models(self, filter_expr: Optional[str] = None) -> List[BaseMLModel]:
        """Retrieve models, optionally filtered."""
        return self.model_accessor.get_data(filter_expr)

    def update_model(self, model: BaseMLModel):
        """Update an existing model."""
        self.model_accessor.update_data_object(model)

    def delete_model(self, model_id: str):
        """Delete a model by its ID."""
        self.model_accessor.delete_data_object(model_id)

    def get_experiment_runs(self, experiment_name: str):
        """Get all runs for a given experiment."""
        experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
        if experiment:
            return self.mlflow_client.search_runs(experiment.experiment_id)
        return []

    def compare_models(self, model_ids: List[str], metric: str):
        """Compare models based on a specific metric."""
        models = [self.load_model(model_id) for model_id in model_ids]
        return {
            model.id: model.model_metadata.get(metric)
            for model in models
            if model.model_metadata
        }

    def get_best_model(self, experiment_name: str, metric: str, mode: str = "max"):
        """Get the best model from an experiment based on a metric."""
        runs = self.get_experiment_runs(experiment_name)
        if not runs:
            return None

        best_run = (
            max(runs, key=lambda run: run.data.metrics.get(metric, float("-inf")))
            if mode == "max"
            else min(runs, key=lambda run: run.data.metrics.get(metric, float("inf")))
        )

        return self.load_model(best_run.data.tags.get("model_id"))
