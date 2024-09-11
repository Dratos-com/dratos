from typing import List, Optional, Dict, Any
import daft
from daft.io import IOConfig
from pyiceberg.table import Table as PyIcebergTable
from beta.data.obj.artifacts.artifact import Artifact
import deltacat
import pyarrow as pa


class ArtifactAccessor:
    """
    Accessor for Artifact objects.
    Provides methods for CRUD operations and data manipulation specific to Artifacts.
    """

    def __init__(self, io_config: IOConfig):
        self.namespace = Artifact.__tablename__
        self.table_name = f"{self.namespace}.artifacts"
        self.io_config = io_config
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Ensure that the Iceberg table exists, create it if it doesn't."""
        try:
            self.pyiceberg_table = PyIcebergTable.load(self.table_name)
        except Exception:
            self._create_table()

    def _create_table(self):
        """Create a new Iceberg table with the schema of the Artifact."""
        schema = Artifact.get_arrow_schema()
        self.pyiceberg_table = PyIcebergTable.create(
            self.table_name,
            schema=schema,
            properties={"write.format.default": "parquet"},
        )

    def get_artifacts(
        self, filter_expr: Optional[daft.Expression] = None
    ) -> List[Artifact]:
        """Retrieve artifacts, optionally filtered by a Daft expression."""
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        df = daft.from_deltacat(deltacat_dataset, io_config=self.io_config)
        if filter_expr is not None:
            df = df.where(filter_expr)
        arrow_table = df.to_arrow()
        return Artifact.from_arrow_table(arrow_table)

    def get_artifact_by_id(self, artifact_id: str) -> Artifact:
        """Retrieve a single artifact by its ID."""
        df = self.get_artifacts(filter_expr=daft.col("id") == artifact_id)
        return Artifact.from_daft_dataframe(df.collect())

    def write_artifacts(self, artifacts: List[Artifact]):
        """Write a list of artifacts to the DeltaCat dataset."""
        arrow_table = Artifact.to_arrow_table(artifacts)
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        deltacat_dataset.write(arrow_table, io_config=self.io_config)

    def update_artifact(self, artifact: Artifact):
        """Update a single artifact in the DeltaCat dataset."""
        df = daft.from_pylist([artifact.dict()])
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        deltacat_dataset.upsert(
            df.to_arrow(), merge_keys=["id"], io_config=self.io_config
        )

    def delete_artifact(self, artifact_id: str):
        """Delete a single artifact from the DeltaCat dataset."""
        df = self.get_artifacts()
        df = df.where(daft.col("id") != artifact_id)
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        deltacat_dataset.write(
            df.to_arrow(), io_config=self.io_config, mode="overwrite"
        )

    def upsert_artifacts(self, artifacts: List[Artifact]):
        """Upsert a list of artifacts to the DeltaCat dataset."""
        df = daft.from_pylist([artifact.dict() for artifact in artifacts])
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        deltacat_dataset.upsert(
            df.to_arrow(), merge_keys=["id"], io_config=self.io_config
        )

    def filter(self, filter_expr: daft.Expression) -> daft.DataFrame:
        """Apply a Daft expression to filter the artifacts."""
        return self.get_artifacts().where(filter_expr)

    def aggregate(
        self, group_by: List[str], agg_exprs: Dict[str, daft.Expression]
    ) -> daft.DataFrame:
        """Perform groupby and aggregation operations on the artifacts."""
        df = self.get_artifacts()
        return df.groupby(group_by).agg(agg_exprs)

    def select(self, *columns: str) -> daft.DataFrame:
        """Select specific columns from the artifacts."""
        return self.get_artifacts().select(*columns)

    def sort(self, by: List[str], ascending: bool = True) -> daft.DataFrame:
        """Sort the artifacts by specified columns."""
        return self.get_artifacts().sort(by, ascending=ascending)

    def limit(self, n: int) -> daft.DataFrame:
        """Limit the number of artifacts."""
        return self.get_artifacts().limit(n)

    def join(
        self, other: daft.DataFrame, on: List[str], how: str = "inner"
    ) -> daft.DataFrame:
        """Join the artifacts with another DataFrame."""
        return self.get_artifacts().join(other, on=on, how=how)

    def create_dataframe(self, data: List[Dict[str, Any]]) -> daft.DataFrame:
        """Create a new Daft DataFrame from a list of dictionaries."""
        return daft.from_pylist(data)

    def sql_query(self, query: str) -> daft.DataFrame:
        """Execute a SQL query on the artifacts."""
        df = self.get_artifacts()
        return df.sql(query)

    def write_dataframe(self, df: daft.DataFrame, mode: str = "append"):
        """Write a Daft DataFrame to the DeltaCat dataset."""
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        deltacat_dataset.write(df.to_arrow(), io_config=self.io_config, mode=mode)

    def show(self, n: int = 10):
        """Display a specified number of artifacts."""
        self.get_artifacts().show(n)

    def to_pandas(self) -> "pandas.DataFrame":
        """Convert the artifacts to a pandas DataFrame."""
        return self.get_artifacts().to_pandas()

    def to_arrow(self) -> "pyarrow.Table":
        """Convert the artifacts to a PyArrow Table."""
        return self.get_artifacts().to_arrow()

    def explain(self):
        """Explain the execution plan for the current artifact retrieval."""
        self.get_artifacts().explain()
