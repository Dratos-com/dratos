from __future__ import annotations

import typing
if typing.TYPE_CHECKING:
    pass
import logging
from enum import Enum
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from ray import serve
from ray.serve.handle import DeploymentHandle
import numpy as np
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from Beta.beta.models.deployments.whisper import whisper_deployment
from Beta.beta.models.deployments.vllm import VLLMDeployment

logger = logging.getLogger("ray.serve")

app = FastAPI()

class AgentStatus(str, Enum):
    INIT = "initializing"
    IDLE = "idle"
    PENDING = "pending"
    WAITING = "waiting"
    PROCESSING = "processing"

class AgentRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    speech: Optional[Union[List[float], List[List[float]]]] = None

class AgentResponse(BaseModel):
    text: str
    status: AgentStatus

@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 5,
        "target_ongoing_requests": 10,
    },
    max_ongoing_requests=20,
)
@serve.ingress(app)
class Agent:
    def __init__(
        self,
        name: str,
        model: DeploymentHandle,
        embedding: DeploymentHandle,
        stt: DeploymentHandle,
        tools: Optional[List[Any]] = None,
        is_async: bool = False,
    ):
        self.name = name
        self.model = model
        self.embedding = embedding
        self.stt = stt
        self.tools = tools or []
        self.is_async = is_async
        self.status = AgentStatus.IDLE

    @app.post("/process")
    async def process(self, request: AgentRequest) -> AgentResponse:
        self.status = AgentStatus.PROCESSING
        try:
            result = await self._process_request(request)
            self.status = AgentStatus.IDLE
            return AgentResponse(text=result, status=self.status)
        except Exception as e:
            self.status = AgentStatus.IDLE
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _process_request(self, request: AgentRequest) -> str:
        transcription = ""
        if request.speech is not None:
            stt_response = await self.stt.remote(request.speech)
            transcription = stt_response[0]  # Assuming the first element is the transcription

        if request.prompt:
            request.prompt += "\n" + transcription
            messages = [{"role": "user", "content": request.prompt}]
        elif request.messages:
            if request.messages[-1]["role"] == "user":
                request.messages[-1]["content"] += " " + transcription
            else:
                request.messages.append({"role": "user", "content": transcription})
            messages = request.messages
        else:
            messages = [{"role": "user", "content": transcription}]

        if not messages:
            raise ValueError("No valid input provided")

        llm_response = await self.model.generate.remote(
                prompt = request.prompt,
                messages = request.messages,
                temperature=0.7,
                max_tokens=100,
            )
        )

        if isinstance(llm_response, ErrorResponse):
            raise ValueError(f"Error from LLM: {llm_response.message}")

        return llm_response.choices[0].message.content

    @app.get("/status")
    def get_status(self) -> AgentStatus:
        return self.status

# Example of how to create and deploy the Agent
def create_agent_deployment(
    name: str,
    model_args: Dict[str, Any],
    embedding_args: Dict[str, Any],
    stt_args: Dict[str, Any],
) -> serve.Deployment:
    vllm_deployment = VLLMDeployment.bind(**model_args)
    whisper_deployment = whisper_deployment.bind(**stt_args)
    # Assuming you have an embedding deployment similar to VLLMDeployment
    embedding_deployment = EmbeddingDeployment.bind(**embedding_args)

    return Agent.bind(
        name=name,
        model=vllm_deployment,
        embedding=embedding_deployment,
        stt=whisper_deployment,
    )

if __name__ == "__main__":
    import ray
    from ray import serve

    ray.init()
    serve.start()

    agent_deployment = create_agent_deployment(
        name="my-agent",
        model_args={
            "model": "NousResearch/Meta-Llama-3-8B-Instruct",
            "tensor_parallel_size": 1,
            "max_num_batched_tokens": 4096,
            "trust_remote_code": True,
        },
        embedding_args={},  # Add embedding model arguments
        stt_args={},  # Add Whisper model arguments
    )

    serve.run(agent_deployment)

    # Example usage
    from ray.serve.handle import DeploymentHandle

    handle = serve.get_deployment("Agent").get_handle()
    response = ray.get(
        handle.process.remote(
            AgentRequest(prompt="What are some highly rated restaurants in San Francisco?")
        )
    )
    print(response)

    serve.shutdown()
    ray.shutdown()