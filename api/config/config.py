from __future__ import annotations

import typing


if typing.TYPE_CHECKING:
    pass

import os
from typing import Optional
import lancedb
import daft
import ray
from openai import AsyncOpenAI, OpenAI
import mlflow
from dynaconf import Dynaconf

from api.config.context.storage_context import StorageContext
from beta.data.obj.base.data_object import DataObject
from api.config.context.client_factory import ClientFactory
from api.config.context.clients.mlflow import MlflowConfig
from beta.data.obj.artifacts.mlflow.mlflow_artifact import MlflowArtifactBridge

# Example (see settings.foo.yaml for full list of options)


class Config(DataObject):
    _instance = None

    def __init__(self, settings: Dynaconf):
        self.settings = Dynaconf(
            settings_files=[
                "settings/settings.dev.yaml",
                "settings/settings.stg.yaml",
            ],
            environments=True,  # Enable environment support
            envvar_prefix=False,  # Prefix for environment variables
            load_dotenv=True,  # Load variables from a .env file
        )

    def load_config(self) -> Config:
        self.storage = StorageContext(self.settings)

    def get_lancedb(
        self,
        storage_context: Optional[StorageContext] = None,
        settings: Optional[Dynaconf] = None,
    ):
        return ClientFactory(storage_context, settings).create_lancedb()

    def get_ray(
        self,
        storage_context: Optional[StorageContext] = None,
        settings: Optional[Dynaconf] = None,
    ):
        return ClientFactory(storage_context, settings).create_ray()

    def get_daft(
        self,
        storage_context: Optional[StorageContext] = None,
        settings: Optional[Dynaconf] = None,
    ):
        return ClientFactory(storage_context, settings).create_daft_client()

    def get_openai(
        self,
        storage_context: Optional[StorageContext] = None,
        settings: Optional[Dynaconf] = None,
    ):
        return ClientFactory(storage_context, settings).create_opeanai_client()

    def get_mlflow(
        self,
        storage_context: Optional[StorageContext] = None,
        settings: Optional[Dynaconf] = None,
    ):
        return ClientFactory(storage_context, settings).create_opeanai_client()

    def get_wandb(
        self,
        storage_context: Optional[StorageContext] = None,
        settings: Optional[Dynaconf] = None,
    ):
        return ClientFactory(storage_context, settings).create_wandb_client()

    def get_triton(
        self,
        storage_context: Optional[StorageContext] = None,
        settings: Optional[Dynaconf] = None,
    ):
        return ClientFactory(storage_context, settings).create_triton_client()


config = Config.get_instance()
