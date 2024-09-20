from abc import ABC, abstractmethod
from typing import Any, Dict
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

class InferenceEngine(ABC):
    @abstractmethod
    def run_inference(self, input_data: Any) -> Any:
        pass

class TensorFlowInferenceEngine(InferenceEngine):
    def __init__(self, model_path: str):
        import tensorflow as tf
        self.model = tf.saved_model.load(model_path)

    def run_inference(self, input_data: Any) -> Any:
        return self.model(input_data)

class ONNXInferenceEngine(InferenceEngine):
    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)

    def run_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        return self.session.run([output_name], {input_name: input_data})[0]

class RayInferenceEngine(InferenceEngine):
    def __init__(self, deployment_handle: DeploymentHandle):
        self.deployment_handle = deployment_handle

    async def run_inference(self, input_data: Any) -> Any:
        try:
            result = await self.deployment_handle.remote(input_data)
            return result
        except Exception as e:
            print(f"Error during Ray inference: {str(e)}")
            raise

@serve.deployment
class RayModelDeployment:
    def __init__(self, model_init_func):
        self.model = model_init_func()

    async def __call__(self, input_data: Any) -> Any:
        return self.model(input_data)

def create_ray_inference_engine(model_init_func, deployment_name: str = "model"):
    RayModelDeployment.options(name=deployment_name).deploy(model_init_func)
    handle = serve.get_deployment(deployment_name).get_handle()
    return RayInferenceEngine(handle)
