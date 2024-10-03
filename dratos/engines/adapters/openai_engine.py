"""
This module provides an OpenAI engine for generating text using the OpenAI API.
"""
import os
import json
import logging
import inspect
from typing import Dict, List, AsyncIterator

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel


from .base_engine import BaseEngine
from dratos.memory.artifacts import Artifact

class OpenAIEngine(BaseEngine):
    """
    OpenAIEngine is a class that wraps the OpenAI API.
    """
    def __init__(
        self,
        api_key: str = os.environ.get("OPENAI_API_KEY"),
        base_url: str = "https://api.openai.com/v1"
    ):
        super().__init__()
        self.api_key = api_key
        is_test_env = os.getenv("IS_TEST_ENV")
        
        if is_test_env == 'true':
            os.environ["ENGINE"] = "OPENAI"
            logging.info("\033[94mTEST ENV SELECTED\033[0m")
            self.base_url = os.getenv("TEST_API_BASE_URL")
        else:
            self.base_url = base_url 
        self.client = None

    async def initialize(self) -> None:
        """
        Initialize the OpenAI engine with the given configuration.
        """
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    async def shutdown(self) -> None:
        """
        Shutdown the OpenAI engine.
        """
        if self.client:
            await self.client.close()

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
        prompt: dict, 
        model_name: str = "gpt-4o",
        output_structure: BaseModel | Dict | None = None,
        tools: List[Dict] = None,
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Generate text from the OpenAI engine.
        """
        if not self.client:
            await self.initialize()

        # Convert structure to a string representation
        if output_structure is not None:
            if isinstance(output_structure, dict):
                structure_str = json.dumps(output_structure)
            elif isinstance(output_structure, BaseModel):
                pass
            else:
                structure_str = str(output_structure)

        # Add the user prompt to the messages
        if messages is None:
        #     if output_structure is None:
        #         messages = [{"role": "system", "content": "You are a helpful assistant"}, 
        #                     {"role": "user", "content": prompt}]
        #     else:
        #         messages = [{"role": "system", "content": "You are a helpful assistant that provides answers in the specified JSON format."}, 
        #                     {"role": "user", "content": prompt}]
        # else:
        #     messages.append({"role": "user", "content": prompt})
            messages = [prompt]
        else:
            messages.append(prompt)

        if output_structure is not None:
            if model_name.startswith("gpt-4o-"):
                response = await self.client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=output_structure,
                    **kwargs,
                ) 
            else:
                response =  await self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        response_format={"type": "json_object"},
                        **kwargs,
                    )
        elif tools is not None:
            response =  await self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    **kwargs,
                )
        else:
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs,
            )
        if not response.choices[0].message.tool_calls:
            result = response.choices[0].message.content
            return result
        else:
            result = {
                "name": response.choices[0].message.tool_calls[0].function.name,
                "arguments": json.loads(response.choices[0].message.tool_calls[0].function.arguments),
                "id": response.choices[0].message.tool_calls[0].id
            }
            return result

    def get_completion_setting(self, **kwargs):
        """
        Get the completion setting for the OpenAI engine.
        """
        if not self.client:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        completion_args = inspect.signature(self.client.chat.completions.create).parameters.keys()
        return list(completion_args)

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

    async def stream(
        self,
        prompt: dict, 
        model_name: str = "gpt-4",
        output_structure: BaseModel | Dict | None = None,
        tools: List[Dict] = None,
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        if not self.client:
            await self.initialize()

        # Prepare messages similar to the generate method
        if messages is None:
            messages = [prompt]
        else:
            messages.append(prompt)

        # Set up the streaming call
        stream = await self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

__all__ = ["OpenAIEngine"]
