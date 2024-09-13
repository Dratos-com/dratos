from typing import Type, List, Optional, Dict, Any
import daft
from daft.io import IOConfig
from pyiceberg.table import Table as PyIcebergTable
from beta.data.obj.data_object import DataObject
import pyarrow as pa
import numpy as np
import lancedb

uri = "data/sample-lancedb"
db = lancedb.connect(uri)
table = db.create_table("my_table",
                         data=[{"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
                               {"vector": [5.9, 26.5], "item": "bar", "price": 20.0}])
result = table.search([100, 100]).limit(2).to_pandas()

class DataObjectTable:
    """
    Table for DataObject subclasses who inherit the __tablename__ and __namespace__ class variables.
    """

    def __init__(self, data_object_class: Type[DataObject]):
        self.data_object_class = data_object_class
        self.namespace = data_object_class.__tablename__
        self.table_name = f"{self.namespace}.{data_object_class.__name__}"


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
    
    def _sorted_bucket_merge_join(self, left: daft.DataFrame, right: daft.DataFrame, on: List[str], how: str = "inner") -> daft.DataFrame:
        """Perform a sorted bucket merge join on the two DataFrames."""
        return left.join(right, on=on, how=how, join_type="sorted_bucket_merge")

    def prep_for_upsert(self, new_data_objects: List[DataObject]) -> daft.DataFrame:
        """
        Prepare data objects for upsert by performing a sorted bucket merge join.
        
        :param new_data_objects: List of new DataObjects to be upserted
        :return: DataFrame ready for upsert
        """
        # Convert new data objects to a Daft DataFrame
        new_df = self.data_object_class.to_daft_dataframe(new_data_objects)
        
        # Get existing data
        existing_df = self.get_data()
        
        # Perform sorted bucket merge join
        merged_df = 
        
        # Coalesce values, preferring the new data
        for column in merged_df.column_names:
            if column != "id":
                merged_df = merged_df.with_column(
                    daft.col(f"{column}_right").fill_null(daft.col(f"{column}_left")).alias(column)
                )
        
        # Select only the relevant columns
        result_df = merged_df.select(self.data_object_class.get_column_names())
        
        return result_df.sort("id")

    def upsert_data_objects(self, new_data_objects: List[DataObject]):
        """
        Upsert new data objects into the existing dataset.
        """
        upsert_df = self.prep_for_upsert(new_data_objects)
        deltacat_dataset = deltacat.Dataset(self.namespace, self.table_name)
        deltacat_dataset.write(upsert_df.to_arrow(), io_config=self.io_config, mode="overwrite")


    

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
