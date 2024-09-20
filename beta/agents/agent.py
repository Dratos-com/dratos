from typing import Protocol, List, Any, Optional
from datetime import datetime
from .result import Result

class ToolInterface(Protocol):
    def execute(self, data: Any) -> Result[Any, Exception]:
        ...

class Agent:
    """
    Orchestrates the execution of models and tools.

    Attributes:
        name (str): The name of the agent.
        models (List[Model]): List of models associated with the agent.
        tools (List[ToolInterface]): List of tools the agent can utilize.
        metadata (Metadata): Metadata information for the agent.
        inference_adapter (Optional[InferenceEngine]): Inference engine adapter.
    """
    def __init__(self, name: str, models: List['Model'], tools: List[ToolInterface], metadata: Metadata, inference_adapter: Optional['InferenceEngine'] = None):
        self.name = name
        self.models = models
        self.tools = tools
        self.metadata = metadata
        self.inference_adapter = inference_adapter

    def execute_pipeline(self, input_data: Any) -> Result[Any, Exception]:
        """
        Executes the agent's processing pipeline.

        Args:
            input_data (Any): The initial input data for the pipeline.

        Returns:
            Result[Any, Exception]: The final output wrapped in a Result monad.
        """
        print(f"Agent {self.name} is executing pipeline.")
        result = Result.Ok(input_data)

        for tool in self.tools:
            result = result.bind(tool.execute)
            if result.is_error:
                print(f"Pipeline halted due to error: {result.value}")
                break

        if self.inference_adapter and not result.is_error:
            try:
                inference_result = self.inference_adapter.run_inference(result.value)
                result = Result.Ok(inference_result)
            except Exception as e:
                result = Result.Error(e)

        return result