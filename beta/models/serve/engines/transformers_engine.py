from typing import Any, Dict, List, Union
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from .base_engine import BaseEngine
from beta.models.serve.utils.prompt_utils import prompt, Prompt
import mlflow


class TransformersEngine(BaseEngine):
    def __init__(
        self,
        model_name: str,
        mlflow: mlflow.tracking.MlflowClient,
        task: str = None,
        **kwargs,
    ):
        super().__init__(model_name, mlflow, **kwargs)
        self.task = task
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._is_initialized = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.task:
            self.pipeline = pipeline(
                self.task, model=self.model_name, device=self.device
            )
        else:
            # Try to load the model based on its architecture
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(
                    self.device
                )
            except:
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name
                    ).to(self.device)
                except:
                    self.model = AutoModel.from_pretrained(self.model_name).to(
                        self.device
                    )

        self._is_initialized = True
        mlflow.transformers.autolog()

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

        with mlflow.start_run(run_name=f"Transformers_{self.model_name}_prediction"):
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("task", self.task)
            mlflow.log_param("input_type", type(input_data).__name__)

            if self.pipeline:
                result = self.pipeline(prompt, **kwargs)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **kwargs)
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            mlflow.log_param("output_type", type(result).__name__)
            mlflow.log_text(str(result), "output.txt")
            mlflow.log_dict(kwargs, "prediction_params.json")

            return result

    async def embed(self, input_data: Union[str, List[str]]) -> torch.Tensor:
        if not self.is_initialized:
            await self.initialize()

        inputs = self.tokenizer(
            input_data, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use the last hidden state as the embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    @property
    def supported_tasks(self) -> List[str]:
        return [
            "text-generation",
            "translation",
            "summarization",
            "question-answering",
            "text-classification",
            "token-classification",
            "zero-shot-classification",
        ]

    def get_model_config(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task": self.task,
            "device": self.device,
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
        if self.tokenizer:
            del self.tokenizer
        if self.pipeline:
            del self.pipeline
        torch.cuda.empty_cache()
        self._is_initialized = False

    def update_config(self, new_config: Dict[str, Any]):
        super().update_config(new_config)
        if self.is_initialized:
            # Reinitialize the model with the new configuration
            self.shutdown()
            self.initialize()


if __name__ == "__main__":
    import asyncio
    import mlflow
    from beta.config import TransformersConfig

    async def main():
        config = TransformersConfig()
        mlflow_client = mlflow.tracking.MlflowClient()
        engine = TransformersEngine(config.model_name, mlflow_client, task=config.task)
        await engine.initialize()

        result = await engine.predict("The future of artificial intelligence is")
        print(result)

        embeddings = await engine.embed(["Hello, world!", "How are you?"])
        print(embeddings)

        await engine.shutdown()

    asyncio.run(main())
