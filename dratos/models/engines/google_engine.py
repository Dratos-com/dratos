"""
This module provides an OpenAI engine for generating text using the OpenAI API.
"""
import os
import json
from urllib.parse import urlparse
from typing import Dict, List, AsyncIterator, Any, Tuple
import base64
import logging

from pydantic import BaseModel

from .base_engine import BaseEngine

logger = logging.getLogger(__name__)

class UnsupportedFeatureError(Exception):
    """Raised when attempting to use features not supported by Gemini."""
    pass

class GoogleEngine(BaseEngine):
    """
    Class that wraps the Google vertex AI API for gemini use.
    """
    def __init__(
        self,
        credentials_path: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT"),
        region: str = os.getenv("GOOGLE_CLOUD_REGION"),
    ):
        
        try:
            from google.oauth2 import service_account
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("google-genai and google-auth are required for GoogleEngine.")

        self.credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self.project_id = project_id
        self.region = region


        super().__init__(engine="OPENAI")
        
    def initialize(self, asynchronous: bool = False) -> None:
        """
        Initialize the OpenAI engine with the given configuration.
        """

        self.client = genai.Client( 
                            credentials=self.credentials,
                            project=self.project_id,
                            location=self.region,
                            vertexai=True
                        )

    def shutdown(self) -> None:
        """
        Shutdown the OpenAI engine.
        """
        if self.client:
            self.client = None

    def sync_gen(
        self,
        model_name: str = "gemini-2.0-flash-001",
        response_model: BaseModel | None = None,
        tools: List[Dict] = None,
        messages: List[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Generate text from the Google engine.
        """
        if not self.client:
            self.initialize(asynchronous=False)

        messages, system_prompt = self.format_messages(messages)

        if response_model is not None:
            config = {
                "automatic_function_calling": {'disable': True, "maximum_remote_calls": None},
                "response_mime_type": "application/json",
                "system_instruction": system_prompt,
                "response_schema": response_model,
                **kwargs,
            }
            response = self.client.models.generate_content(
                model=model_name, 
                contents=messages,
                config=config,
            )
        elif tools is not None:
            raise NotImplementedError("Tool calling is not implemented for Gemini.")
        else:
            config = {
                "automatic_function_calling": {'disable': True, "maximum_remote_calls": None},
                "system_instruction": system_prompt,
                **kwargs,
            }
            response = self.client.models.generate_content(
                model=model_name, 
                contents=messages,
                config=config,
            )

        if not response.function_calls:
            result = response.text
            return result
        else:
            # TODO: Implement tool calling
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
        model_name: str = "gemini-2.0-flash-001",
        messages: List[Dict] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        
        if not self.client:
            self.initialize(asynchronous=True)

        messages, system_prompt = self.format_messages(messages)

        config = {
            "automatic_function_calling": {'disable': True, "maximum_remote_calls": None},
            "system_instruction": system_prompt,
            **kwargs,
        }

        for chunk in self.client.models.generate_content_stream(
            model = model_name,
            contents = messages,
            config = config,
        ):
            yield chunk.text

    def format_messages(self, messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """
        Converts internal message format to Gemini's content format using Part system.
        
        Args:
            messages: List of internal message dictionaries
            
        Returns:
            Tuple of (formatted messages list, system prompt)
            
        Raises:
            UnsupportedFeatureError: When encountering unsupported message types
            ValueError: For unknown message roles or invalid formats
        """
        formatted_messages = []
        system_prompt = None
        
        for message in messages:
            role = message["role"]
            
            if role == "System prompt":
                system_prompt = message["content"]["text"]
                
            elif role == "Prompt":
                parts = []
                content = message["content"]
                
                # Add text content if present
                if "text" in content:
                    parts.append(types.Part.from_text(text=content["text"]))
                
                # Handle media content (images and PDFs)
                for key, value in content.items():
                    if key == "text":
                        continue
                        
                    extension = key.split('.')[-1].lower()
                    if extension not in ['png', 'jpg', 'jpeg', 'webp', 'pdf']:
                        continue
                        
                    mime_type = f"image/{extension}" if extension in ['png', 'jpg', 'jpeg', 'webp'] else "application/pdf"
                    
                    if isinstance(value, str):
                        if urlparse(value).scheme in ["http", "https"]:
                            # Handle URLs
                            parts.append(types.Part.from_uri(
                                file_uri=value,
                                mime_type=mime_type
                            ))
                        elif value.startswith('data:'):
                            # Handle data URLs
                            try:
                                data = value.split(',')[1]
                                parts.append(types.Part.from_bytes(
                                    data=base64.b64decode(data),
                                    mime_type=mime_type
                                ))
                            except IndexError:
                                logger.error(f"Invalid data URL for {key}")
                        else:
                            # Handle raw base64 data
                            try:
                                parts.append(types.Part.from_bytes(
                                    data=base64.b64decode(value),
                                    mime_type=mime_type
                                ))
                            except Exception as e:
                                logger.error(f"Failed to decode base64 data for {key}: {e}")
                
                formatted_messages.append({
                    "role": "user",
                    "parts": parts
                })
                
            elif role == "Response":
                formatted_messages.append({
                    "role": "model",
                    "parts": [types.Part.from_text(text=message["content"]["text"])]
                })
                
            elif role == "Tool call":
                raise UnsupportedFeatureError(
                    "Tool/Function calling is not supported by Gemini."
                )
                
            else:
                raise ValueError(f"Unknown message role: {role}")
        
        return formatted_messages, system_prompt
    
    def support_tools(self, model_name: str) -> bool:
        # TODO: Add support for tools
        return False
    
    def support_structured_output(self, model_name: str) -> bool:
        return True
    
    def supported_documents(self, model_name: str, extension: str) -> bool:
        # TODO: Add support for other document types (video, audio, etc.)
        return extension in [".png", ".jpeg", ".pdf", ".webp"]

