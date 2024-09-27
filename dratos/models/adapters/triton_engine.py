from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
import numpy as np
import requests
import tritonserver
from fastapi import FastAPI
from PIL import Image
from ray import serve
from dataclasses import dataclass

app = FastAPI()


@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class TritonServerEngine:
    def __init__(
        self,
        config,
    ):
        model_repository = config.TRITON_MODEL_REPO

        self._triton_server = tritonserver.Server(
            model_repository=model_repository,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
        )
        self._triton_server.start(wait_until_ready=True)

    @app.get("/generate")
    def generate(self, prompt: str, filename: str = "generated_image.jpg") -> str:
        if not self._triton_server.model("stable_diffusion").ready():
            try:
                self._triton_server.load("text_encoder")
                self._triton_server.load("vae")
                self._stable_diffusion = self._triton_server.load("stable_diffusion")
                if not self._stable_diffusion.ready():
                    raise Exception("Model not ready")
            except Exception as error:
                return f"Error loading stable diffusion model: {error}"

        for response in self._stable_diffusion.infer(inputs={"prompt": [[prompt]]}):
            generated_image = (
                np.from_dlpack(response.outputs["generated_image"])
                .squeeze()
                .astype(np.uint8)
            )

            image = Image.fromarray(generated_image)
            image.save(filename)
            return f"Image generated and saved as {filename}"


if __name__ == "__main__":
    from dratos.config import TritonConfig

    # Initialize and run the TritonServerEngine
    config = TritonConfig()
    engine = TritonServerEngine(config)

    # Generate an image
    result = engine.generate(
        prompt="A serene landscape with mountains and a lake",
        filename="generated_landscape.jpg",
    )
    print(result)

# Usage example
"""
# Initialize and run the TritonServerEngine
config = Config(TRITON_MODEL_REPO="/path/to/model/repository")
engine = TritonServerEngine(config)

# Generate an image
result = engine.generate(
    prompt="A serene landscape with mountains and a lake",
    filename="generated_landscape.jpg"
)
print(result)
"""
