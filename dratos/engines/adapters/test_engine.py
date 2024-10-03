import json
import os
from typing import Dict, List, AsyncIterator
from pydantic import BaseModel
from .base_engine import BaseEngine
from dratos.memory import Artifact

class TestEngine(BaseEngine):
    """
    TestEngine is a class that mimics OpenAIEngine for testing purposes.
    It uses a JSON file to store test cases instead of calling the OpenAI API.
    """
    def __init__(
        self,
        test_cases_file: str = "test_cases.json",
        base_url: str = "https://api.openai.com/v1"
    ):
        super().__init__()
        self.test_cases_file = test_cases_file
        self.base_url = base_url
        self.test_cases = {}

    async def initialize(self) -> None:
        """
        Initialize the TestEngine by loading test cases from the JSON file.
        """
        if os.path.exists(self.test_cases_file):
            with open(self.test_cases_file, 'r') as f:
                self.test_cases = json.load(f)
        else:
            self.test_cases = {}

    async def shutdown(self) -> None:
        """
        Shutdown the TestEngine. No action needed for this implementation.
        """
        pass

    async def get_supported_models(self):
        """
        Get the supported models for the TestEngine.
        """
        return list(self.test_cases.keys())

    async def generate(
        self,
        prompt: dict, 
        model_name: str = "gpt-4",
        output_structure: BaseModel | Dict | None = None,
        tools: List[Dict] = None,
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Generate text from the TestEngine using predefined test cases.
        """
        if messages is None:
            messages = [prompt]
        else:
            messages.append(prompt)

        # Create a unique key for the test case
        test_case_key = json.dumps({
            "model": model_name,
            "messages": messages,
            "output_structure": str(output_structure) if output_structure else None,
            "tools": tools,
            **kwargs
        })

        if test_case_key in self.test_cases:
            result = self.test_cases[test_case_key]
            if isinstance(result, dict) and "tool_calls" in result:
                return {
                    "name": result["tool_calls"][0]["function"]["name"],
                    "arguments": json.loads(result["tool_calls"][0]["function"]["arguments"]),
                    "id": result["tool_calls"][0]["id"]
                }
            else:
                return result
        else:
            raise ValueError(f"No test case found for the given input: {test_case_key}")

    def get_completion_setting(self, **kwargs):
        """
        Get the completion setting for the TestEngine.
        """
        # Return a predefined list of settings
        return ["model", "messages", "temperature", "max_tokens", "n", "stop", "presence_penalty", "frequency_penalty"]

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using predefined test cases.
        """
        if "embeddings" in self.test_cases:
            return self.test_cases["embeddings"].get(json.dumps(texts), [[0.0] * 4096 for _ in texts])
        else:
            return [[0.0] * 4096 for _ in texts]

    async def get_embedding(self, text: str) -> List[float]:
        """
        Get the embedding for a single text using predefined test cases.
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0]

    @property
    def supported_tasks(self) -> List[str]:
        """
        Get the supported tasks for the TestEngine.
        """
        return ["text-generation", "chat", "structured-generation"]
    
    def get_supported_tasks(self) -> List[str]:
        """
        Get the supported tasks for the TestEngine.
        """
        return self.supported_tasks

    async def log_artifacts(self, artifacts: List[Artifact]) -> None:
        """
        Log the artifacts for the TestEngine. This is a no-op for testing purposes.
        """
        pass

    async def stream(
        self,
        prompt: dict, 
        model_name: str = "gpt-4",
        output_structure: BaseModel | Dict | None = None,
        tools: List[Dict] = None,
        messages: List[Dict[str, str]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream text from the TestEngine using predefined test cases.
        """
        full_response = await self.generate(prompt, model_name, output_structure, tools, messages, **kwargs)
        if isinstance(full_response, str):
            # Simulate streaming by yielding one word at a time
            for word in full_response.split():
                yield word + " "
        else:
            # If it's not a string (e.g., a dict for tool calls), yield it as-is
            yield json.dumps(full_response)

    def add_test_case(self, input_data: dict, output_data: str | dict):
        """
        Add a new test case to the JSON file.
        """
        test_case_key = json.dumps(input_data)
        self.test_cases[test_case_key] = output_data
        with open(self.test_cases_file, 'w') as f:
            json.dump(self.test_cases, f, indent=2)

__all__ = ["TestEngine"]