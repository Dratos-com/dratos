import lancedb
import daft
import ray
import openai
import mlflow

from api.config.resources.storage_config import StorageConfig
from api.config.clients.lancedb import LanceDBConfig
from api.config.clients.daft import DaftConfig

class ClientFactory:
    def __init__(self, storage_config: StorageConfig, settings: Dynaconf):
        self.storage_config = storage_config
        self.settings = settings

    def create_lancedb_client(self):
        lance_config = LanceDBConfig(sc=self.storage_config, settings=self.settings)

        db = lancedb.connect(
            f"s3://{lance_config.storage_options.bucket_name}",
            storage_options=lance_config.storage_options,
        )
        return db

    def get_daft(self) -> daft:
        daft_config = DaftConfig(sc = self.storage_config, settings=self.settings)
        
        daft.set_execution_config(
            scan_tasks_min_size_bytes = daft_config.execution.scan_tasks_min_size_bytes,
            scan_tasks_max_size_bytes: int | None = None,
            broadcast_join_size_bytes_threshold: int | None = None,
            parquet_split_row_groups_max_files: int | None = None,
            sort_merge_join_sort_with_aligned_boundaries: bool | None = None,
            hash_join_partition_size_leniency: bool | None = None,
            sample_size_for_sort: int | None = None,
            num_preview_rows: int | None = None,
            parquet_target_filesize: int | None = None,
            parquet_target_row_group_size: int | None = None,
            parquet_inflation_factor: float | None = None,
            csv_target_filesize: int | None = None,
            csv_inflation_factor: float | None = None,
            shuffle_aggregation_default_partitions: int | None = None,
            read_sql_partition_size_bytes: int | None = None,
            enable_aqe: bool | None = None,
            enable_native_executor: bool | None = None,
            default_morsel_size: int | None = None
        )
            
        daft.set_planning_config(default_io_config=DaftConfig.storage)

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
