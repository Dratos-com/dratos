import os
from typing import List, Optional
from outlines import models, generate

from beta.data.structs.models import Prompt, PromptSettings, Message
from api.config import config


model = models.llamacpp(
    "NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF",
    "Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf",
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
        "NousResearch/Hermes-2-Pro-Llama-3-8B"
    ),
    n_gpu_layers=-1,
    flash_attn=True,
    n_ctx=8192,
    verbose=False,
)

model = outlines.models.transformers("TheBloke/Mistral-7B-OpenOrca-AWQ", device="cuda")


@serve.deployment(
    name="vllm",
    num_replicas=2,
    ray_actor_options={"num_cpus": 0.2, "num_gpus": 0},
    max_ongoing_requests=100,
    health_check_period_s=10,
    health_check_timeout_s=30,
    graceful_shutdown_timeout_s=20,
    graceful_shutdown_wait_loop_s=2,
)
class LLM:
    """Base Multimodal Language Model that defines the interface for all language models"""
    def __init__(self,
        model_name: str = "NousResearch/llama-3.1-8b-instruct",
        is_async: bool = True,
    ):

    async def __call__(
        self,
        prompt: Prompt,
        messages: Optional[List[Message]],
        response_model: Optional[DomainObject],
    ) -> str:
        """Chat with the model"""

        client = self.get_client()
        client.api_key = self.api_key

        mlflow.openai.autolog()
        response = client.chat.completions.create(
            response_format=response_model,
            model=self.model_name,
            messages=messages,
            max_tokens=self.prompt_settings.max_tokens,
            temperature=self.prompt_settings.temperature,
            top_p=self.prompt_settings.top_p,
            seed=self.prompt_settings.seed,
            stream=self.is_async,
        )
        return response

    def get_client(self) -> OpenAI | AsyncOpenAI:
        client = config.config.openai_proxy(
            is_async=self.is_async, api_key=self.api_key
        )

        return client


llm_deployment = LLM.bind()
