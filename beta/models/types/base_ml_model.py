""" define a base class for all machine learning models """
from abc import ABC, abstractmethod


class BaseMLModel(ABC):
    """
    Abstract base class for all machine learning models.
    """
    @abstractmethod
    def predict(self, input_data):
        """
        Abstract method for making predictions.
        
        Args:
            input_data: The input data for prediction.
        
        Returns:
            Prediction result.
        """
        pass

    @abstractmethod
    def train(self, training_data):
        """
        Abstract method for training the model.
        
        Args:
            training_data: The data to train the model on.
        """
        pass

    # Add any other common methods or properties here
