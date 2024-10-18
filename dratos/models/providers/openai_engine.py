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
        logging.info("\033[94mTEST ENV: \033[0m", is_test_env)
        
        if is_test_env == 'true':
            os.environ["ENGINE"] = "OPENAI"
            logging.info("\033[94mTEST ENV SELECTED\033[0m")
            self.base_url = os.getenv("TEST_API_BASE_URL")
            self.api_key = "TEST_API_KEY"
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

    async def generate(
        self,
        prompt: dict, 
        model_name: str = "gpt-4o",
        output_structure: BaseModel | str | None = None,
        tools: List[Dict] = None,
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Generate text from the OpenAI engine.
        """
        if not self.client:
            await self.initialize()

        if messages is None:
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
