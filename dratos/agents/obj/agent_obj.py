from __future__ import annotations

import typing
if typing.TYPE_CHECKING:
    pass
from enum import Enum
from typing import List, Optional
import daft
from pydantic import Field
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse
import numpy as np

from dratos.models.deployments.whisper import whisper_deployment

class AgentStatus(str, Enum):
    INIT = "initializing"
    IDLE = "idle"
    PENDING = "pending"
    WAITING = "waiting"
    PROCESSING = "processing"

class Agent:
    """
    The base agent class.
    """
    def __init__(self,
        name: str = Field(..., description="The name of the agent"),
        model: DeploymentHandle = Field(..., description="Ray Serve Deployment Handler Reference to Model Deployment"),
        embedding: DeploymentHandle = Field(..., description="Ray Serve DeploymentHandler to Embedding Model Deployment"),
        stt: DeploymentHandle = Field(..., description="Ray Serve DeploymentHandler to Speech To Text Model Deployment"),
        tools: Optional[List[Tool]] = Field(default=None, description="The tools that the agent can use"),
        is_async: bool = Field(default=False, description="Use asynchrony (i.e. for streaming)."),
        ):
        self.model = model
        self.tokenizer = Autotokenizer
        self.stt = whisper_delployment

        self.status = 

    async def __call__(self, 
        prompt: Optional[Prompt], 
        messages: Optional[List[Message]], 
        speech: Optional[Union[np.ndarray | List[float] | List[np.ndarray] | List[List[float]]]] = None,
        ) -> str:

        if speech is not None: 
            stt_response: DeploymentResponse = self.stt.remote(speech)
        
        

        llm_response: DeploymentResponse = self.model.generate.remote(Prompt)


        return self.model.generate(
                prompt=prompt,
                messages=messages,
                response_model=tool,
                device="cuda"
            )
    

    def generate(self,re **kwargs):
        

        response = self.model.generate(
            **kwargs,
        ).remote(request)
        return result.get()

    def transcribe(self):
        self.stt.remote(request)
        
    async def __call__(self, input: int) -> int:
        adder_response: DeploymentResponse = self._adder.remote(input)
        # Pass the adder response directly into the multipler (no `await` needed).
        multiplier_response: DeploymentResponse = self._multiplier.remote(
            adder_response
        )
        # `await` the final chained response.
        return await multiplier_response

    def get_status(self) -> AgentStatus:
        return self.status
    
    def infer_action(self, prompt: Prompt, actions: List[str], tools: List[Tool]) -> str:
        """
        Infer the action to perform based on the prompt and tools.
        """
        return "python"
    
    def choose_tool(self, tools: List[Tool]) -> Tool:
        """
        Choose the tool to use based on the tools.
        """
        # TODO: Implement tool selection logic
        return Tool(
            name="python",
            desc="Python tool",
            type=ToolTypes.python,
            function=lambda x: x + 1
        )
    

agent_deployment = Agent.bind(
    llm_deployment,
    whisper_deployment
    )

__all__ = ["Agent"]
