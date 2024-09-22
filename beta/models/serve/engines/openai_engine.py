"""
This module provides an OpenAI engine for generating text using the OpenAI API.
"""
import os
import logging
from typing import Any, Coroutine, Dict, List, Union, Optional
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
import daft
import json
import lark
import pyarrow as pa
from .base_engine import BaseEngine, BaseEngineConfig
from ....data import Artifact
from ....data import DataObject
from outlines.fsm.json_schema import build_regex_from_schema
from outlines.processors.base_logits_processor import OutlinesLogitsProcessor
from outlines.processors.structured import JSONLogitsProcessor
from outlines.processors.structured import RegexLogitsProcessor
import numpy as np

class OpenAIEngineConfig(BaseEngineConfig):
    """
    Configuration for the OpenAI engine.
    """
    def __init__(self, data: Dict[str, Any] = None):
        """
        Initialize the OpenAI engine configuration.
        """
        super().__init__()
        if data is None:
            data = {}

        # Convert single values to lists
        self._data = {k: [v] if not isinstance(v, (list, tuple)) else v for k, v in data.items()}
        self.df = daft.from_pydict(self._data)

    def update(self, new_config: Dict[str, Any]) -> None:
        """
        Update the OpenAI engine configuration with new data.
        """
        new_data = {k: [v] if not isinstance(v, (list, tuple)) else v for k, v in new_config.items()}
        self._data.update(new_data)
        self.df = daft.from_pydict(self._data)

    def get(self, name, default=None):
        """
        Get the value of a configuration option.
        """
        if name in self._data:
            return self._data[name][0]
        return default

    def __getattr__(self, name):
        """
        Get the value of a configuration option.
        """
        return self.get(name)


class OpenAIEngine(BaseEngine):
    """
    OpenAIEngine is a class that wraps the OpenAI API.
    """
    def __init__(
        self,
        config: dict = OpenAIEngineConfig()
    ):
        super().__init__(config=config)
        self.config: OpenAIEngineConfig = config
        self.client: Optional[Union[OpenAI | AsyncOpenAI]] = AsyncOpenAI(
            api_key=config.get("api_key", os.environ.get("OPENAI_API_KEY")),
            base_url=config.get("base_url", "https://api.openai.com/v1")
        )
        self.model_name = config.get("model_name", "gpt-4o")
        self.embedding_model = "text-embedding-ada-002"

    @property
    def get_model_name(self):
        return self.config.get("model_name", "gpt-4o")
    
    @get_model_name.setter
    def set_model_name(self, value):
        self.model_name = value
        self.config.update({"model_name": value})

    def set_logits_processor(self, processor: OutlinesLogitsProcessor) -> None:
        """
        Set the logits processor for the OpenAI engine.
        """
        self.logits_processor = processor

    def get_logits_processor(self) -> OutlinesLogitsProcessor:
        """
        Get the logits processor for the OpenAI engine.
        """
        return self.logits_processor

    async def generate_structured(self, prompt: str | List[str], structure: str | Dict | Any, messages: Optional[List[Dict[str, str]]]=None, config: Optional[OpenAIEngineConfig]=None, **kwargs) -> Coroutine[Any, Any, Any]:
        """
        Generate structured data from the OpenAI engine.
        """
        if not self.client:
            await self.initialize(config if config else self.config)

        if config: 
            self.config = config
            self.model_name = config.get("model_name", "gpt-4o")

        if isinstance(prompt, list):
            prompt = "\n".join(prompt)

        # Convert structure to a string representation
        if isinstance(structure, dict):
            structure_str = json.dumps(structure)
        elif isinstance(structure, pa.Schema):
            structure_str = structure.to_string()
        else:
            structure_str = str(structure)

        # Add the structure requirement to the prompt
        json_prompt = f"{prompt}\nPlease provide the answer in the following JSON structure: {structure_str}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides answers in the specified JSON format."},
            {"role": "user", "content": json_prompt}
        ]

        response_format = {"type": "json_object"}

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format=response_format,
            **kwargs
        )

        result = response.choices[0].message.content
        return result

    async def initialize(self, config: OpenAIEngineConfig) -> None:
        """
        Initialize the OpenAI engine with the given configuration.
        """
        self.config = config
        self.client = AsyncOpenAI(
            api_key=self.config.get("api_key"),
            base_url=self.config.get("base_url", "https://api.openai.com/v1")
        )
        # Get the first available model for embeddings
        models = await self.client.models.list()
        self.embedding_model = "text-embedding-ada-002"

    async def shutdown(self) -> None:
        """
        Shutdown the OpenAI engine.
        """
        if self.client:
            await self.client.close()

    async def get_model_info(self):
        """
        Get the model information for the OpenAI engine.
        """
        model = self.client.models.retrieve(self.model_name)
        return model

    async def get_supported_models(self):
        """
        Get the supported models for the OpenAI engine.
        """
        if not self.client:
            await self.initialize(self.config)
        models = await self.client.models.list()
        return [model.id for model in models.data]

    async def generate(
        self,
        prompt: str,
        messages: List[Dict[str, str]] = [{"role": "system", "content": "You are a helpful assistant"}],
        response_format: BaseModel = None,
        **kwargs,
    ):
        """
        Generate text from the OpenAI engine.
        """
        if not self.client:
            await self.initialize(self.config)

        # Add the user prompt to the messages
        if messages is not None:
            messages.append({"role": "user", "content": prompt})
        else:
            messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}]

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format=response_format,
            **kwargs,
        )
        result = response.choices[0].message.content

        return result

    async def generate_data(
        self,
        prompt: Union[str, List[str]],
        structure: Union[str, Dict, pa.Schema, DataObject],
        grammar: lark.Lark = None,
        logits_processor: Optional[OutlinesLogitsProcessor] = None,
        **kwargs,
    ):
        """
        Generate data from the OpenAI engine.
        """
        if logits_processor is None:
            logits_processor = self.get_logits_processor(structure=structure)
        else:
            self.logits_processor(logits_processor)
        result = await self.generate(prompt, **kwargs)

        return result

    async def get_logits_processor(
        self, structure: Union[str, dict, pa.Schema, BaseModel]
    ):
        """
        Get the logits processor for the given structure.
        """
        if isinstance(structure, str):
            self.set_logits_processor(RegexLogitsProcessor(structure))
        elif isinstance(structure, dict):
            regex_str = build_regex_from_schema(structure)
            self.set_logits_processor(
                RegexLogitsProcessor(regex_str, self.client.tokenizer)
            )
        elif isinstance(structure, pa.Schema):
            json_schema = structure
            schema_str = convert_json_schema_to_str(json_schema)
            regex_str = build_regex_from_schema(schema_str)
            self.set_logits_processor(
                RegexLogitsProcessor(regex_str, self.client.tokenizer)
            )
        elif issubclass(structure, BaseModel):
            json_schema = structure.model_json_schema()
            schema_str = str(json_schema)
            regex_str = build_regex_from_schema(schema_str)
            self.logits_processor = RegexLogitsProcessor(regex_str, self.client.models)

        else:
            raise ValueError("Unsupported structure type")

        return self.logits_processor

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using the vLLM API.

        Args:
            texts (List[str]): A list of texts to generate embeddings for.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        try:
            response = await self.client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logging.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 4096 for _ in texts]

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for a single text.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            List[float]: The embedding as a list of floats.
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0]

    @property
    def supported_tasks(self) -> List[str]:
        """
        Get the supported tasks for the OpenAI engine.
        """
        return ["text-generation", "chat", "structured-generation"]
    
    def get_supported_tasks(self) -> List[str]:
        """
        Get the supported tasks for the OpenAI engine.
        """
        return self.supported_tasks

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the model configuration for the OpenAI engine.
        """
        return {
            "model_name": self.model_name,
            "api_key": "***",  # Masked for security
            **self.config.dict(),
        }

    async def log_artifacts(self, artifacts: List[Artifact]) -> None:
        """
        Log the artifacts for the OpenAI engine.
        """
        for artifact in artifacts:
            if self.mlflow_client:
                self.mlflow_client.log_artifact(
                    run_id=self.mlflow_client.active_run().info.run_id,
                    local_path=artifact.path,
                )


__all__ = ["OpenAIEngine", "OpenAIEngineConfig"]
