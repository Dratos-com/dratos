from abc import ABC, abstractmethod
from typing import Any

class InferenceEngine(ABC):
    @abstractmethod
    def run_inference(self, input_data: Any) -> Any:
        pass

class TensorFlowInferenceEngine(InferenceEngine):
    def run_inference(self, input_data: Any) -> Any:
        # TensorFlow-specific inference logic
        ...

class ONNXInferenceEngine(InferenceEngine):
    def run_inference(self, input_data: Any) -> Any:
        # ONNX-specific inference logic
        ...