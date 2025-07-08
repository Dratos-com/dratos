"""
This module provides an OpenAI engine for generating text using the OpenAI API.
"""
import os
import json
from urllib.parse import urlparse
from typing import Dict, List, AsyncIterator, Any

from pydantic import BaseModel

from .base_engine import BaseEngine


class OpenAIEngine(BaseEngine):
    """
    OpenAI is a class that wraps the OpenAI API.
    """
    OpenAI = None
    AsyncOpenAI = None

    def __init__(
        self,
        api_key: str = os.getenv("OPENAI_API_KEY"),
        base_url: str = "https://api.openai.com/v1"
    ):
        self.api_key = api_key
        self.base_url = base_url

        super().__init__(engine="OPENAI")
        
    def initialize(self, asynchronous: bool = False) -> None:
        """
        Initialize the OpenAI engine with the given configuration.
        """

        try:
            from openai import OpenAI, AsyncOpenAI
            OpenAIEngine.OpenAI = OpenAI
            OpenAIEngine.AsyncOpenAI = AsyncOpenAI
        except ImportError:
            raise ImportError("openai is required for OpenAIEngine.")

        if asynchronous:
            self.client = OpenAIEngine.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = OpenAIEngine.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    def shutdown(self) -> None:
        """
        Shutdown the OpenAI engine.
        """
        if self.client:
            self.client.close()
            self.client = None

    def sync_gen(
        self,
        model_name: str = "gpt-4o",
        response_model: BaseModel | None = None,
        tools: List[Dict] = None,
        messages: List[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Generate text from the OpenAI engine.
        """
        if not self.client:
            self.initialize(asynchronous=False)

        messages = self.format_messages(messages)

        if response_model is not None:
            if model_name.startswith("gpt-4o-"):
                response = self.client.beta.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=response_model,
                    **kwargs,
                )
            else:
                response =  self.client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        **kwargs,
                    )
        elif tools is not None:
            response =  self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    **kwargs,
                )
        else:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **kwargs,
            )
        if not response.choices[0].message.tool_calls:
            result = response.choices[0].message.content
            return result
        else:
            result = [
                {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                    "id": tool_call.id
                }
                for tool_call in response.choices[0].message.tool_calls
            ]
            return result

    async def async_gen(
        self,
        model_name: str = "gpt-4",
        response_model: BaseModel | None = None,
        tools: List[Dict] = None,
        messages: List[Dict] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate text from the OpenAI engine asynchronously with streaming.
        Supports tools and structured output by accumulating the response and post-processing.
        """
        
        if not self.client:
            self.initialize(asynchronous=True)

        messages = self.format_messages(messages)

        # Prepare completion arguments
        completion_kwargs = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        # Handle structured output
        if response_model is not None:
            if model_name.startswith("gpt-4o-"):
                # Use beta.chat.completions.parse for structured output
                stream = await self.client.beta.chat.completions.parse(
                    response_format=response_model,
                    **completion_kwargs
                )
            else:
                # Fallback to regular completion
                stream = await self.client.chat.completions.create(**completion_kwargs)
        elif tools is not None:
            # Add tools to completion
            completion_kwargs["tools"] = tools
            completion_kwargs["tool_choice"] = "auto"
            stream = await self.client.chat.completions.create(**completion_kwargs)
        else:
            # Standard streaming
            stream = await self.client.chat.completions.create(**completion_kwargs)

        # Accumulate response for tools/structured output processing
        if tools is not None or response_model is not None:
            accumulated_content = ""
            tool_calls_data = None
            
            # Collect all chunks
            async for chunk in stream:
                if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                    # Handle tool calls
                    if tool_calls_data is None:
                        tool_calls_data = []
                    
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        # Extend tool_calls_data list if needed
                        while len(tool_calls_data) <= tool_call.index:
                            tool_calls_data.append({"id": "", "function": {"name": "", "arguments": ""}})
                        
                        if tool_call.id:
                            tool_calls_data[tool_call.index]["id"] = tool_call.id
                        if tool_call.function.name:
                            tool_calls_data[tool_call.index]["function"]["name"] = tool_call.function.name
                        if tool_call.function.arguments:
                            tool_calls_data[tool_call.index]["function"]["arguments"] += tool_call.function.arguments
                
                elif chunk.choices[0].delta.content is not None:
                    content_chunk = chunk.choices[0].delta.content
                    accumulated_content += content_chunk
                    yield content_chunk
            
            # Process accumulated response for tools
            if tool_calls_data:
                # Return tool calls as the final result
                result = [
                    {
                        "name": tool_call["function"]["name"],
                        "arguments": json.loads(tool_call["function"]["arguments"]),
                        "id": tool_call["id"]
                    }
                    for tool_call in tool_calls_data
                ]
                # For async streaming with tools, we yield the final result as JSON
                yield json.dumps(result)
            elif response_model is not None and accumulated_content:
                # For structured output, the accumulated content should already be formatted
                # The response_model validation will be handled at the Agent level
                pass  # Content was already yielded during streaming
        else:
            # Standard streaming without tools or structured output
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted_messages = []
        for message in messages:
            if message["role"] == "System prompt":
                formatted_messages.append({"role": "system", "content": message["content"]["text"]})
            elif message["role"] == "Prompt":
                content = [{"type": "text", "text": message["content"]["text"]}]
                for key, value in message["content"].items():
                    if key != "text" and (key.endswith(".png") or key.endswith(".jpg") or key.endswith(".jpeg") or key.endswith(".gif") or key.endswith(".webp")):
                        if isinstance(value, str) and urlparse(value).scheme in ["http", "https", "ftp", "ftps", "data"]:
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": value
                                }
                            })
                        else:
                            extension = key.split('.')[-1]
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{extension};base64,{value}"
                                }
                            })
                formatted_messages.append({"role": "user", "content": content})
            elif message["role"] == "Response":
                formatted_messages.append({"role": "assistant", "content": message["content"]["text"]})
            elif message["role"] == "Tool call":
                content = [
                    {
                    "type": "function",
                    "id": message["context"]["id"],
                    "function": {
                        "name": message["content"]["name"],
                        "arguments": f'{message["content"]["arguments"]}'
                        }
                    }
                ]
                formatted_messages.append({"role": "assistant", "tool_calls": content})
            else:
                raise ValueError(f"Unknown message role: {message['role']}")
        return formatted_messages
    
    def support_tools(self, model_name: str) -> bool:
        return True
    
    def support_structured_output(self, model_name: str) -> bool:
        if model_name.startswith("gpt-4o-"):
            return True
        return False
    
    def supported_documents(self, model_name: str, extension: str) -> bool:
        if model_name == "gpt-4o-mini":
            return extension in [".png", ".docx", ".pdf", ".txt"]
        return False
