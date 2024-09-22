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
from api.config.context.client_factory import MlflowConfig
from beta.data.obj.artifacts.mlflow.mlflow_artifact import MlflowArtifactBridge

# Example (see settings.foo.yaml for full list of options)


class Config(DataObject):
    _instance = None

    @property
    def get_storage(self):
        return StorageContext(self.settings)
    
    @property
    def get_settings(self):
        return Dynaconf(
            settings_files=[
                "/teamspace/studios/this_studio/beta/api/settings/settings.dev.yaml"
            ],
            load_dotenv=False
        )
    def __init__(self, settings: Dynaconf):
        self.settings = Dynaconf(
            settings_files=[
                "/teamspace/studios/this_studio/beta/api/settings/settings.dev.yaml"
            ],
            load_dotenv=False
        )

    def load_config(self) -> Config:
        self.storage = StorageContext(self.settings)
        return self
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
        return ClientFactory(storage_context, settings).get_daft()

    def get_openai(
        self,
        storage_context: Optional[StorageContext] = None,
        settings: Optional[Dynaconf] = None,
    ):
        return ClientFactory(storage_context, settings).get_openai()

    def get_mlflow(
        self,
        storage_context: Optional[StorageContext] = None,
        settings: Optional[Dynaconf] = None,
    ):
        return ClientFactory(storage_context, settings).get_mlflow()

    def get_wandb(
        self,
        storage_context: Optional[StorageContext] = None,
        settings: Optional[Dynaconf] = None,
    ):
        return ClientFactory(storage_context, settings).get_wandb()

    def get_triton(
        self,
        storage_context: Optional[StorageContext] = None,
        settings: Optional[Dynaconf] = None,
    ):
        return ClientFactory(storage_context, settings).get_triton()

    def get_instance():
        if Config._instance is None:
            Config._instance = Config(settings=Dynaconf(settings_files=["/teamspace/studios/this_studio/beta/api/settings/settings.dev.yaml"]))
        return Config._instance


config = Config.get_instance()


class LanceDBContext(StorageContext):
    def __init__(self, settings: Dynaconf):
        super().__init__(settings)
        self.settings = settings

    def get_lancedb(self):
        return ClientFactory(self, self.settings).create_lancedb()

