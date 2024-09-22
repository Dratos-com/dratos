import lancedb
import daft
from dynaconf import Dynaconf
import ray
import openai
import mlflow
import os

from api.config.context.storage_context import StorageContext
from api.config.clients.lancedb import LanceDBConfig
from api.config.clients.daft import DaftLocalConfig as DaftConfig
from api.config.clients.mlflow import MlflowConfig

from beta.data.obj.artifacts.mlflow.mlflow_artifact import MlflowArtifactBridge


class ClientFactory:
    def __init__(self, storage_config: StorageContext, settings: Dynaconf):
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
        daft_config = DaftConfig(sc=self.storage_config)

        daft.set_execution_config(daft_config.config)

        daft.set_planning_config()

    # MLFlow Client
    def get_mlflow(self):
        mlflow_config = MlflowConfig(sc=self.storage_config, settings=self.settings)

        mlflow.set_tracking_uri(self.settings.mlflow.tracking_uri)
        mlflow.set_registry_uri(mlflow_config.storage)

    def get_ray(self):
        ray_config = RayConfig(sc=self.storage_config, settings=self.settings)

        ray.init(address=ray_config.address)

    def create_ray(self):
        ray_config = self.settings.get('dev.config.clients.ray', {})
        address = ray_config.get('address', 'auto')
        local_mode = ray_config.get('local_mode', False)

        ray.init(address=address, local_mode=local_mode)
        return ray
    

class RayConfig:
    def __init__(self, sc: StorageContext, settings: Dynaconf):
        self.sc = sc
        self.settings = settings

    def get_address(self):
        return self.settings.ray.address
    
    @property
    def address(self):
        return self.settings.ray.address

