import os
from typing import Any, Dict, List
from llama_cpp import Llama
from .base_engine import BaseEngine
from dratos.models.serve.utils.prompt_utils import prompt
import mlflow


class LlamaCppEngine(BaseEngine):
    def __init__(self, model_name: str, mlflow: mlflow.tracking.MlflowClient, **kwargs):
        super().__init__(model_name, mlflow, **kwargs)
        self.model = None
        self._is_initialized = False

    async def initialize(self):
        model_path = os.path.join("models", f"{self.model_name}.bin")
        self.model = Llama(model_path=model_path, **self.config)
        self._is_initialized = True
        mlflow.llama_cpp.autolog()  # Enable LlamaCpp autologging

    @prompt
    def generate_prompt(self, input_data: Any) -> str:
        """
        {{ input_data }}
        """

    async def predict(self, input_data: Any, **kwargs) -> Any:
        if not self.is_initialized:
            await self.initialize()

        prompt_instance = self.generate_prompt
        prompt = prompt_instance(input_data)

        with mlflow.start_run(run_name=f"LlamaCpp_{self.model_name}_prediction"):
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("input_type", type(input_data).__name__)

            result = self.model(prompt, **kwargs)
            mlflow.log_param("output_type", "text")
            mlflow.log_text(result["choices"][0]["text"], "output.txt")

            mlflow.log_dict(kwargs, "prediction_params.json")

            return result["choices"][0]["text"]

    @property
    def supported_tasks(self) -> List[str]:
        return ["text-generation", "chat"]

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            **self.config,
        }

    def register_model(self):
        mlflow.log_param("registered_model", self.model_name)

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def shutdown(self):
        if self.model:
            del self.model
        self._is_initialized = False

    def update_config(self, new_config: Dict[str, Any]):
        """Update the engine configuration."""
        self.config.update(new_config)

    def usage_example(self):
        """
        Provides a usage example for the LlamaCppEngine.
        This method demonstrates how to initialize the engine, make a prediction,
        and shut it down.
        """


if __name__ == "__main__":
    import asyncio

    # Create an instance of LlamaCppEngine
    engine = LlamaCppEngine(
        model_name="llama-7b", mlflow=mlflow.tracking.MlflowClient()
    )

    async def run_example():
        # Initialize the engine
        await engine.initialize()

        # Prepare input data
        input_data = "What is the capital of France?"

        # Make a prediction
        result = await engine.predict(input_data, max_tokens=50)
        print(f"Input: {input_data}")
        print(f"Output: {result}")

    # Run the example
    asyncio.run(run_example())

    engine.shutdown()
