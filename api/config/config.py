from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
import os
from dotenv import load_dotenv
from typing import Optional
import httpx

import daft
from daft.io import IOConfig, S3Config, GCSConfig, AzureConfig
import ray
from openai import AsyncOpenAI, OpenAI

# from unitycatalog import AsyncUnitycatalog, DefaultHttpxClient
import mlflow
import yaml


class Config:
    _instance = None

    def __init__(self, is_async: bool = True):
        load_dotenv()
        self.load_config("config.yaml")
        




    def load_config(self, config_file: str) -> Config:
        

    def get_services(self):
        self.get_daft()
        self.get_lancedb()
        self.get_mlflow()
        self.get_ray()
   
    def get_storage_context(self):
        # Override config with environment variables
        # if AWS_ACCESS_KEY_ID is set, assume we are using S3
        
            



    def get_ray(self) -> ray:

        s3_config = config_dict.get('storage', {}).get('s3', {})
        s3_config['access_key_id'] = os.getenv('AWS_ACCESS_KEY_ID', s3_config.get('access_key_id'))
        s3_config['secret_access_key'] = os.getenv('AWS_SECRET_ACCESS_KEY', s3_config.get('secret_access_key'))
        ray.init(self.RAY_RUNNER_HEAD, runtime_env={"pip": ["getdaft"]})
        return ray

    def get_daft(self) -> daft:
        return daft

    def get_mlflow(self) -> mlflow:
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        return mlflow

    def get_openai_proxy(
        self,
        engine: str = "openai",
        is_async: Optional[bool] = None,
    ) -> AsyncOpenAI | OpenAI:
        if is_async is None:
            is_async = self.is_async
        return self.get_client(engine)

    def get_triton(self) -> tritonserver.Server:
        return tritonserver.Server(
            model_repository=self.TRITON_MODEL_REPO,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
        )

    def deploy_autosave_worker(self):
        script_content = """
        addEventListener('fetch', event => {
          event.respondWith(handleRequest(event.request))
        })

        async function handleRequest(request) {
          if (request.method === 'POST') {
            const data = await request.json()
            // Here you would typically encrypt the data with the user's key
            // For demonstration, we're just echoing it back
            return new Response(JSON.stringify(data), {
              headers: { 'Content-Type': 'application/json' }
            })
          }
          return new Response('Send a POST request with JSON data to autosave', { status: 200 })
        }
        """
        return self.deploy_to_cloudflare("autosave-worker", script_content)


config = Config.get_instance()


if __name__ == "__main__":
    # Usage example
    config = Config.get_instance()

    # Initialize Ray
    ray = config.get_ray()
    print("Ray initialized:", ray.is_initialized())

    # Get Daft
    daft = config.get_daft()
    print("Daft version:", daft.__version__)

    # Set up MLflow
    mlflow = config.get_mlflow()
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    # Get OpenAI proxy
    openai_async = config.get_openai_proxy(engine="openai", is_async=True)
    print("OpenAI async client:", type(openai_async))

    openai_sync = config.get_openai_proxy(engine="openai", is_async=False)
    print("OpenAI sync client:", type(openai_sync))

    # Initialize Triton server
    triton_server = config.get_triton()
    print("Triton server model repository:", triton_server.model_repository)

    # Deploy autosave worker (Note: This will actually deploy to Cloudflare if configured)
    # worker = config.deploy_autosave_worker()
    # print("Autosave worker deployed:", worker)

    # Clean up
    ray.shutdown()
    print("Ray shut down")
