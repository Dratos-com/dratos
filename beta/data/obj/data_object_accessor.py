from typing import Type, List, Optional, Dict, Any
import daft
from daft.io import IOConfig
from pyiceberg.table import Table as PyIcebergTable
import deltacat
from beta.data.obj.data_object import DataObject
import pyarrow as pa
import numpy as np


class DataAccessor:
    """
    Generic accessor for DataObject subclasses.
    Provides methods for CRUD operations and data manipulation using Daft and DeltaCat.
    """

    def __init__(self, data_object_class: Type[DataObject], io_config: IOConfig):
        self.data_object_class = data_object_class
        self.namespace = data_object_class.__tablename__
        self.table_name = f"{self.namespace}.{data_object_class.__name__}"
        self.io_config = io_config
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Ensure that the Iceberg table exists, create it if it doesn't."""
        try:
            self.pyiceberg_table = PyIcebergTable.load(self.table_name)
        except Exception:
            self._create_table()

    def _create_table(self):
        """Create a new Iceberg table with the schema of the DataObject."""
        schema = self.data_object_class.get_arrow_schema()
        self.pyiceberg_table = PyIcebergTable.create(
            self.table_name,
            schema=schema,
            properties={"write.format.default": "parquet"},
        )

    def get_data(
        self, filter_expr: Optional[daft.Expression] = None
    ) -> List[DataObject]:
        """Retrieve data objects, optionally filtered by a Daft expression."""
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        df = daft.from_deltacat(deltacat_dataset, io_config=self.io_config)
        if filter_expr is not None:
            df = df.where(filter_expr)
        arrow_table = df.to_arrow()
        return self.data_object_class.from_arrow_table(arrow_table)

    def get_data_object_by_id(self, object_id: str) -> DataObject:
        """Retrieve a single data object by its ID."""
        df = self.get_data(filter_expr=daft.col("id") == object_id)
        return self.data_object_class.from_daft_dataframe(df.collect())

    def write_data_objects(self, data_objects: List[DataObject]):
        """Write a list of data objects to the DeltaCat dataset."""
        arrow_table = self.data_object_class.to_arrow_table(data_objects)
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        deltacat_dataset.write(arrow_table, io_config=self.io_config)

    def update_data_object(self, data_object: DataObject):
        """Update a single data object in the DeltaCat dataset."""
        df = daft.from_pylist([data_object.dict()])
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        deltacat_dataset.upsert(
            df.to_arrow(), merge_keys=["id"], io_config=self.io_config
        )

    def delete_data_object(self, object_id: str):
        """Delete a single data object from the DeltaCat dataset."""
        df = self.get_data()
        df = df.where(daft.col("id") != object_id)
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        deltacat_dataset.write(
            df.to_arrow(), io_config=self.io_config, mode="overwrite"
        )

    def filter(self, filter_expr: daft.Expression) -> daft.DataFrame:
        """Filter the data using a Daft expression."""
        return self.get_data().where(filter_expr)

    def aggregate(
        self, group_by: List[str], agg_exprs: Dict[str, daft.Expression]
    ) -> daft.DataFrame:
        """Perform groupby and aggregation operations."""
        df = self.get_data()
        return df.groupby(group_by).agg(agg_exprs)

    def select(self, *columns: str) -> daft.DataFrame:
        """Select specific columns from the data."""
        return self.get_data().select(*columns)

    def sort(self, by: List[str], ascending: bool = True) -> daft.DataFrame:
        """Sort the data by specified columns."""
        return self.get_data().sort(by, ascending=ascending)

    def limit(self, n: int) -> daft.DataFrame:
        """Limit the number of rows in the data."""
        return self.get_data().limit(n)

    def join(
        self, other: daft.DataFrame, on: List[str], how: str = "inner"
    ) -> daft.DataFrame:
        """Join the data with another DataFrame."""
        return self.get_data().join(other, on=on, how=how)

    def create_dataframe(self, data: List[Dict[str, Any]]) -> daft.DataFrame:
        """Create a new Daft DataFrame from a list of dictionaries."""
        return daft.from_pylist(data)

    def sql_query(self, query: str) -> daft.DataFrame:
        """Execute a SQL query on the data."""
        df = self.get_data()
        return df.sql(query)

    def write_dataframe(self, df: daft.DataFrame, mode: str = "append"):
        """Write a Daft DataFrame to the DeltaCat dataset."""
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        deltacat_dataset.write(df.to_arrow(), io_config=self.io_config, mode=mode)

    def show(self, n: int = 10):
        """Display a specified number of rows from the data."""
        self.get_data().show(n)

    def to_pandas(self) -> "pandas.DataFrame":
        """Convert the data to a pandas DataFrame."""
        return self.get_data().to_pandas()

    def to_arrow(self) -> "pyarrow.Table":
        """Convert the data to a PyArrow Table."""
        return self.get_data().to_arrow()

    def explain(self):
        """Explain the execution plan for the current data retrieval."""
        self.get_data().explain()
