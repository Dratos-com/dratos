import os
from typing import Any, Dict, List
import openai
from pydantic import BaseModel
from .base_engine import BaseEngine
from dratos.engines.serve.utils.prompt_utils import prompt, Prompt
import mlflow
from dratos.memory.obj.base.data_object import DataObject


class OpenRouterEngine(BaseEngine):
    def __init__(self, model_name: str, mlflow: mlflow.tracking.MlflowClient, **kwargs):
        super().__init__(model_name, mlflow, **kwargs)
        self.client = None
        self._is_initialized = False
        self.base_url = "https://openrouter.ai/api/v1"

    async def initialize(self):
        openai.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = openai.AsyncOpenAI(base_url=self.base_url)
        self._is_initialized = True
        mlflow.openai.autolog()  # Enable OpenAI autologging (works for OpenRouter too)

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

        with mlflow.start_run(run_name=f"OpenRouter_{self.model_name}_prediction"):
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("input_type", type(input_data).__name__)

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            result = response.choices[0].message.content
            mlflow.log_param("output_type", "text")
            mlflow.log_text(result, "output.txt")

            mlflow.log_dict(kwargs, "prediction_params.json")

            return result

    @property
    def supported_tasks(self) -> List[str]:
        return ["text-generation", "chat"]

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "api_key": "***",  # Masked for security
            "base_url": self.base_url,
            **self.config,
        }

    def register_model(self):
        mlflow.log_param("registered_model", self.model_name)

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def shutdown(self):
        self._is_initialized = False

    # The update_config method is inherited from BaseEngine
