# Example of how to create and deploy the Agent
from typing import Any, Dict


from api.deployments.agents.SpeechAgent import AgentRequest
from api.deployments.vllm.vllm_deployment import VLLMDeployment
from api.deployments.embeddings.embedding_deployment import EmbeddingDeployment
from api.deployments.whisper.whisper_deployment import WhisperDeployment

from beta import Agents, Data, Models, Tools

from 
from beta.agents import deployments
from beta.agents.obj. import Agent

from beta.models.engines import BaseEngine, OpenAIEngine


def create_agent_deployment(
    name: str,
    model_args: Dict[str, Any],
    embedding_args: Dict[str, Any],
    stt_args: Dict[str, Any],
) -> deployments:
    vllm_deployment = VLLMDeployment.bind(**model_args)
    whisper_deployment = WhisperDeployment.bind(**stt_args)
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

    _deployment = create_agent_deployment(
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