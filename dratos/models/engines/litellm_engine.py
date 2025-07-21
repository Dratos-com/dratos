"""
This module provides a LiteLLM engine for generating text using multiple LLM providers through LiteLLM's unified API.
"""
import os
import json
from urllib.parse import urlparse
from typing import Dict, List, AsyncIterator, Any

from pydantic import BaseModel

from .base_engine import BaseEngine


class LiteLLMEngine(BaseEngine):
    """
    LiteLLM engine that provides access to 100+ LLM providers through a unified interface.
    Supports OpenAI, Anthropic, Google, Azure OpenAI, Cohere, and many other providers.
    """
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        provider: str = "openai",
        # Vertex AI specific parameters - using same env var names as GoogleEngine
        vertex_project: str = None,
        vertex_location: str = None,
        vertex_credentials: str = None,
        **provider_kwargs
    ):
        """
        Initialize LiteLLM engine.
        
        Args:
            api_key: API key for the provider (if not set via environment)
            base_url: Custom base URL (for custom endpoints)
            provider: Provider to use (openai, anthropic, azure, vertex_ai, etc.)
            vertex_project: Google Cloud project ID for Vertex AI (uses GOOGLE_CLOUD_PROJECT env var)
            vertex_location: Google Cloud region for Vertex AI (uses GOOGLE_CLOUD_REGION env var)
            vertex_credentials: Path to service account JSON file (uses GOOGLE_APPLICATION_CREDENTIALS env var)
            **provider_kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider
        
        # Vertex AI specific configuration - defaults from environment variables
        # Use same env var names as GoogleEngine for consistency
        self.vertex_project = vertex_project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.vertex_location = vertex_location or os.getenv("GOOGLE_CLOUD_REGION")
        self.vertex_credentials = vertex_credentials or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        self.provider_kwargs = provider_kwargs
        
        # Store provider-specific configuration
        self._setup_provider_config()
        
        super().__init__(engine="LITELLM")
        
    def _setup_provider_config(self):
        """Setup provider-specific configuration via environment variables."""
        # LiteLLM uses environment variables for different providers
        if self.api_key:
            if self.provider == "openai":
                os.environ["OPENAI_API_KEY"] = self.api_key
            elif self.provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = self.api_key
            elif self.provider == "azure":
                os.environ["AZURE_API_KEY"] = self.api_key
            elif self.provider == "gemini":
                os.environ["GEMINI_API_KEY"] = self.api_key
            # Add more providers as needed
            
        if self.base_url:
            if self.provider == "openai":
                os.environ["OPENAI_BASE_URL"] = self.base_url
            elif self.provider == "azure":
                os.environ["AZURE_API_BASE"] = self.base_url
                
        # Setup Vertex AI configuration using same env vars as GoogleEngine
        if self.provider == "vertex_ai" or any(param for param in [self.vertex_project, self.vertex_location, self.vertex_credentials]):
            if self.vertex_project:
                # Set both LiteLLM and Google Cloud standard env vars
                os.environ["VERTEXAI_PROJECT"] = self.vertex_project
                os.environ["GOOGLE_CLOUD_PROJECT"] = self.vertex_project
            elif "VERTEXAI_PROJECT" not in os.environ and "VERTEX_PROJECT" not in os.environ:
                # Try to get from Google Cloud standard env vars (consistent with GoogleEngine)
                if "GOOGLE_CLOUD_PROJECT" in os.environ:
                    os.environ["VERTEXAI_PROJECT"] = os.environ["GOOGLE_CLOUD_PROJECT"]
                elif "GCP_PROJECT" in os.environ:
                    os.environ["VERTEXAI_PROJECT"] = os.environ["GCP_PROJECT"]
                elif "PROJECT_ID" in os.environ:
                    os.environ["VERTEXAI_PROJECT"] = os.environ["PROJECT_ID"]
                        
            if self.vertex_location:
                # Set both LiteLLM and Google Cloud standard env vars
                os.environ["VERTEXAI_LOCATION"] = self.vertex_location
                os.environ["GOOGLE_CLOUD_REGION"] = self.vertex_location
            elif "VERTEXAI_LOCATION" not in os.environ and "VERTEX_LOCATION" not in os.environ:
                # Try to get from Google Cloud standard env vars (consistent with GoogleEngine)
                if "GOOGLE_CLOUD_REGION" in os.environ:
                    os.environ["VERTEXAI_LOCATION"] = os.environ["GOOGLE_CLOUD_REGION"]
                else:
                    # Default to us-central1 if not specified
                    os.environ["VERTEXAI_LOCATION"] = "us-central1"
                    os.environ["GOOGLE_CLOUD_REGION"] = "us-central1"
                
            if self.vertex_credentials:
                # Handle both file paths and JSON strings
                if os.path.isfile(self.vertex_credentials):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.vertex_credentials
                else:
                    # Assume it's a JSON string, save to temp file
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        if isinstance(self.vertex_credentials, str):
                            try:
                                # Try to parse as JSON string
                                cred_dict = json.loads(self.vertex_credentials)
                                json.dump(cred_dict, f, indent=2)
                            except json.JSONDecodeError:
                                # If not JSON, assume it's already a path
                                raise ValueError(f"Invalid vertex_credentials: {self.vertex_credentials}")
                        else:
                            json.dump(self.vertex_credentials, f, indent=2)
                        temp_path = f.name
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
            elif "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
                # Try to use same default behavior as GoogleEngine
                pass  # Let LiteLLM handle authentication via gcloud auth
                
    def initialize(self, asynchronous: bool = False) -> None:
        """
        Initialize the LiteLLM engine.
        
        Args:
            asynchronous: Whether to use async client (LiteLLM handles this automatically)
        """
        try:
            import litellm
            self.litellm = litellm
            
            # Configure LiteLLM settings
            if self.base_url and self.provider == "openai":
                litellm.api_base = self.base_url
                
            # Set up any additional provider kwargs
            for key, value in self.provider_kwargs.items():
                setattr(litellm, key, value)
                
        except ImportError:
            raise ImportError("litellm is required for LiteLLMEngine. Install with: pip install litellm")

    def shutdown(self) -> None:
        """
        Shutdown the LiteLLM engine and close any HTTP sessions.
        """
        pass

    def sync_gen(
        self,
        model_name: str = "gpt-4o",
        response_model: BaseModel | None = None,
        tools: List[Dict] = None,
        messages: List[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Generate text from the LiteLLM engine synchronously.
        
        Args:
            model_name: Model to use (can include provider prefix like 'anthropic/claude-3-sonnet')
            response_model: Pydantic model for structured output
            tools: List of tool/function definitions
            messages: Conversation messages
            **kwargs: Additional completion parameters
        """
        if not hasattr(self, 'litellm'):
            self.initialize(asynchronous=False)

        # Format messages to OpenAI format (LiteLLM uses OpenAI format internally)
        formatted_messages = self.format_messages(messages)
        
        # Prepare completion arguments
        completion_kwargs = {
            "model": model_name,
            "messages": formatted_messages,
            **kwargs
        }
        
        # Handle structured output
        if response_model is not None:
            completion_kwargs["response_format"] = response_model
            
        # Handle tools
        if tools is not None:
            completion_kwargs["tools"] = tools
            completion_kwargs["tool_choice"] = "auto"
            
        # Add Vertex AI specific parameters if using vertex_ai provider
        if self.provider == "vertex_ai" or model_name.startswith("vertex_ai/"):
            if self.vertex_project:
                completion_kwargs["vertex_project"] = self.vertex_project
            if self.vertex_location:
                completion_kwargs["vertex_location"] = self.vertex_location
            if self.vertex_credentials and os.path.isfile(self.vertex_credentials):
                completion_kwargs["vertex_credentials"] = self.vertex_credentials

        # Make the completion call
        response = self.litellm.completion(**completion_kwargs)
        
        # Process response
        if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
            # Return tool calls
            result = [
                {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                    "id": tool_call.id
                }
                for tool_call in response.choices[0].message.tool_calls
            ]
            return result
        else:
            # Return text content
            return response.choices[0].message.content

    async def async_gen(
        self,
        model_name: str = "gpt-4",
        response_model: BaseModel | None = None,
        tools: List[Dict] = None,
        messages: List[Dict] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate text from the LiteLLM engine asynchronously with streaming.
        Supports tools and structured output by accumulating the response and post-processing.
        
        Args:
            model_name: Model to use
            response_model: Pydantic model for structured output
            tools: List of tool/function definitions
            messages: Conversation messages
            **kwargs: Additional completion parameters
        """
        if not hasattr(self, 'litellm'):
            self.initialize(asynchronous=True)

        # Format messages
        formatted_messages = self.format_messages(messages)
        
        # Prepare completion arguments
        completion_kwargs = {
            "model": model_name,
            "messages": formatted_messages,
            "stream": True,
            **kwargs
        }
        
        # Handle structured output
        if response_model is not None:
            completion_kwargs["response_format"] = response_model
            
        # Handle tools
        if tools is not None:
            completion_kwargs["tools"] = tools
            completion_kwargs["tool_choice"] = "auto"
        
        # Add Vertex AI specific parameters if using vertex_ai provider
        if self.provider == "vertex_ai" or model_name.startswith("vertex_ai/"):
            if self.vertex_project:
                completion_kwargs["vertex_project"] = self.vertex_project
            if self.vertex_location:
                completion_kwargs["vertex_location"] = self.vertex_location
            if self.vertex_credentials and os.path.isfile(self.vertex_credentials):
                completion_kwargs["vertex_credentials"] = self.vertex_credentials
        
        # Make async streaming completion call
        stream = await self.litellm.acompletion(**completion_kwargs)

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
        """
        Format messages from internal format to OpenAI format (used by LiteLLM).
        This is identical to OpenAI engine since LiteLLM uses OpenAI format internally.
        """
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
        """
        Check if the model supports tools/function calling.
        LiteLLM supports tools for most major providers including Vertex AI.
        """
        # Models that support tools through LiteLLM
        if any(prefix in model_name.lower() for prefix in [
            'gpt-', 'claude-', 'gemini-', 'vertex_ai/gemini-', 'vertex_ai/claude-'
        ]):
            return True
        
        # Vertex AI models generally support tools
        if model_name.startswith('vertex_ai/'):
            return True
            
        return True  # Default to True since LiteLLM handles compatibility
    
    def support_structured_output(self, model_name: str) -> bool:
        """
        Check if the model supports structured output (JSON schema).
        """
        # LiteLLM supports structured output for compatible models
        supported_models = [
            'gpt-4o', 'claude-3', 'gemini-', 'vertex_ai/gemini-', 
            'vertex_ai/claude-', 'gemini-1.5', 'gemini-2.0'
        ]
        
        if any(prefix in model_name.lower() for prefix in supported_models):
            return True
        return False
    
    def supported_documents(self, model_name: str, extension: str) -> bool:
        """
        Check if the model supports a specific document type.
        """
        # Vision/document support varies by model
        supported_extensions = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".txt", ".docx"]
        
        # Check if model supports vision/documents
        vision_models = [
            'gpt-4o', 'gpt-4-vision', 'claude-3', 'gemini-pro-vision', 
            'gemini-2', 'gemini-1.5', 'vertex_ai/gemini-', 'vertex_ai/claude-3'
        ]
        
        if any(vision_indicator in model_name.lower() for vision_indicator in vision_models):
            return extension.lower() in supported_extensions
            
        return False 