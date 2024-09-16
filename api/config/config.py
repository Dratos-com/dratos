from __future__ import annotations

import typing


if typing.TYPE_CHECKING:
    pass

import os
from typing import Optional
import lancedb
import daft
from daft.io import IOConfig, S3Config, GCSConfig, AzureConfig

import ray
from openai import AsyncOpenAI, OpenAI
import mlflow
from dynaconf import Dynaconf

from api.config.resources.storage_config import StorageConfig
from beta.data.obj.data_object import DataObject
from api.config.resources.client_factory import ClientFactory

# Example (see settings.foo.yaml for full list of options)


class Config(DataObject):
    _instance = None

    def __init__(self, settings: Dynaconf):
        self.settings = Dynaconf(
            settings_files=[
                "settings.dev.yaml",
                "settings.prod.yaml",
            ],
            environments=True,  # Enable environment support
            envvar_prefix="DRATOS",  # Prefix for environment variables
            load_dotenv=True,  # Load variables from a .env file
        )

    def load_config(self) -> Config:
        self.storage = StorageConfig(self.settings)

    def get_lancedb(self):
        lancedb_client = ClientFactory(
            self.storage, self.settings
        ).create_lancedb_client()
        return lancedb_client

    def get_ray(self):
        return ClientFactory(self.storage, self.settings).get_ray()

    def get_daft(self):
        return ClientFactory(self.storage, self.settings).get_daft()

    def get_openai_proxy(self):
        return ClientFactory(self.storage, self.settings).get_openai_proxy()

    def get_mlflow(self):
        return ClientFactory(self.storage, self.settings).get_mlflow()

    def get_triton(self):
        return ClientFactory(self.storage, self.settings).get_triton()


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
