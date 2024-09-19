import lancedb
import daft
import ray
import openai
import mlflow
import os

from api.config.context.storage_context import StorageConfig
from api.config.clients.lancedb import LanceDBConfig
from api.config.clients.daft import DaftConfig
from api.config.clients.mlflow import MlflowConfig

from beta.data.obj.artifacts.mlflow.mlflow_artifact import MlflowArtifactBridge


class ClientFactory:
    def __init__(self, storage_config: StorageConfig, settings: Dynaconf):
        self.storage_config = storage_config
        self.settings = settings

    # LanceDB Client
    def create_lancedb(self):
        lance_config = LanceDBConfig(
            storage_context=self.storage_config, settings=self.settings
        )

        db = lancedb.connect(
            f"s3://{lance_config.storage_options.bucket_name}",
            storage_options=lance_config.storage_options,
        )
        return db

    # Daft Client
    def get_daft(self) -> daft:
        daft_config = DaftConfig(sc=self.storage_config, settings=self.settings)

        daft.set_execution_config(daft_config)

        daft.set_planning_config(default_io_config=DaftConfig.storage)

    # MLFlow Client
    def get_mlflow(self):
        mlflow_config = MlflowConfig(sc=self.storage_config, settings=self.settings)

        mlflow.set_tracking_uri()
        mlflow.set_registry_uri(mlflow_config.storage)
