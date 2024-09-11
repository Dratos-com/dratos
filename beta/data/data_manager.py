from beta.services.model_manager import ModelManager
from beta.data.obj.data_object import DataObject
from typing import List, Dict, Any
from beta.data.obj.base import DataObjectManager


class DataManager:
    def __init__(self, dataset_uri: str, experiment_name: str):
        # ... (existing initialization)
        self.model_manager = ModelManager(dataset_uri, experiment_name)

    def train_and_save_model(
        self,
        model: BaseMLModel,
        training_data: List[DataObject],
        model_params: Dict[str, Any],
    ):
        # Train the model (this is just a placeholder, implement your actual training logic)
        model.train(training_data)

        # Save the model and log parameters
        self.model_manager.save_model(model, model_params)

    def make_predictions(self, model_id: str, input_data: List[DataObject]):
        model = self.model_manager.load_model(model_id)
        predictions = model.predict(input_data)
        self.model_manager.log_predictions(input_data, predictions)
        return predictions

    def get_best_model(self, metric: str):
        return self.model_manager.get_best_model(self.experiment_name, metric)

    # ... (other data management methods)
