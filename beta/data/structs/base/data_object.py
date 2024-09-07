from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from typing import Optional, List, Dict, Any, ClassVar, Type, TypeVar
import pyarrow as pa
from datetime import datetime
import ulid
from lancedb.pydantic import pydantic_to_schema
import deltacat


class DataObjectError(Exception):
    """Base exception class for CustomModel errors."""


class ULIDValidationError(DataObjectError):
    """Raised when ULID validation fails."""


T = TypeVar("T", bound="DataObject")


@dataclass
class DataObject(BaseModel):
    __tablename__: ClassVar[str] = "data"

    id: str = Field(
        default_factory=lambda: ulid.ulid(), description="ULID Unique identifier"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="UTC timestamp of last update"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )
    type: ClassVar[str] = "DomainObject"
    arrow_schema: pa.Schema = Field(
        description="An Apache Arrow Schema generated from the pydantic class with the cls.to_arrow() method"
    )

    class Config:
        allowed = True

    @property
    def created_at(self) -> datetime:
        return ulid.parse(self.id).timestamp().datetime

    def update_timestamp(self):
        self.updated_at = datetime.now(datetime.timezone.utc)

    @classmethod
    def create_new(cls: Type[T], **data) -> T:
        return cls(
            id=ulid.new().str, updated_at=datetime.now(datetime.timezone.utc), **data
        )

    @classmethod
    def to_arrow(cls) -> pa.Schema:
        return pydantic_to_schema(cls)

    def to_arrow_record_batch(self) -> pa.RecordBatch:
        schema = self.to_arrow()
        data = [pa.array([getattr(self, field.name)]) for field in schema]
        return pa.RecordBatch.from_arrays(data, schema=schema)


class DataAccessor:
    def __init__(self, table_name: str):
        self.table_name = table_name
        self._dataset = None

    @property
    def dataset(self):
        if self._dataset is None:
            try:
                from deltacat import Dataset
            except ImportError:
                raise ImportError("deltacat must be installed to use this method.")
            self._dataset = Dataset(self.table_name)
        return self._dataset

    @classmethod
    async def get_data(
        cls,
        table_name: str,
        partition_keys: Optional[Dict[str, Any]] = None,
        filters: Optional[List[Dict[str, Any]]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> daft.DataFrame:
        """
        Retrieve data as a Daft DataFrame using DeltaCat.

        Args:
            table_name (str): The name of the table to query.
            partition_keys (Optional[Dict[str, Any]]): Partition keys to filter the data.
            filters (Optional[List[Dict[str, Any]]]): Additional filters to apply.
            columns (Optional[List[str]]): Specific columns to retrieve.
            limit (Optional[int]): Maximum number of rows to return.

        Returns:
            daft.DataFrame: A Daft DataFrame containing the requested data.
        """
        try:
            import daft
            from deltacat import Dataset
        except ImportError:
            raise ImportError("daft and deltacat must be installed to use this method.")

        dataset = Dataset(table_name)

        # Apply partition keys if provided
        if partition_keys:
            for key, value in partition_keys.items():
                dataset = dataset.with_partition(key, value)

        # Create a Daft DataFrame from the DeltaCat Dataset
        df = daft.from_deltacat(dataset)

        # Apply filters if provided
        if filters:
            for filter_dict in filters:
                column = filter_dict.get("column")
                op = filter_dict.get("op")
                value = filter_dict.get("value")
                if column and op and value is not None:
                    df = df.filter(getattr(df[column], op)(value))

        # Select specific columns if provided
        if columns:
            df = df.select(columns)

        # Apply limit if provided
        if limit:
            df = df.limit(limit)

        return df

    def to_pandas(self, table_name: str, **kwargs) -> pd.DataFrame:
        """
        Retrieve data as a Pandas DataFrame.

        Args:
            table_name (str): The name of the table to query.
            **kwargs: Additional arguments to pass to the retrieve_data method.

        Returns:
            pd.DataFrame: A Pandas DataFrame containing the requested data.
        """
        daft_df = self.retrieve_data(table_name, **kwargs)
        return daft_df.to_pandas()

    def to_arrow(self, table_name: str, **kwargs) -> pa.Table:
        """
        Retrieve data as a PyArrow Table.

        Args:
            table_name (str): The name of the table to query.
            **kwargs: Additional arguments to pass to the retrieve_data method.

        Returns:
            pa.Table: A PyArrow Table containing the requested data.
        """
        daft_df = self.retrieve_data(table_name, **kwargs)
        return daft_df.to_arrow()

    def to_parquet(self, table_name: str, file_path: str, **kwargs) -> None:
        """
        Retrieve data and save it as a Parquet file.

        Args:
            table_name (str): The name of the table to query.
            file_path (str): The path where the Parquet file will be saved.
            **kwargs: Additional arguments to pass to the retrieve_data method.
        """
        daft_df = self.retrieve_data(table_name, **kwargs)
        daft_df.to_parquet(file_path)

    def to_csv(self, table_name: str, file_path: str, **kwargs) -> None:
        """
        Retrieve data and save it as a CSV file.

        Args:
            table_name (str): The name of the table to query.
            file_path (str): The path where the CSV file will be saved.
            **kwargs: Additional arguments to pass to the retrieve_data method.
        """
        daft_df = self.retrieve_data(table_name, **kwargs)
        daft_df.to_csv(file_path)

    def get_schema(self, table_name: str) -> pa.Schema:
        """
        Retrieve the schema of a table.

        Args:
            table_name (str): The name of the table to query.

        Returns:
            pa.Schema: The PyArrow schema of the table.
        """
        try:
            from deltacat import Dataset
        except ImportError:
            raise ImportError("deltacat must be installed to use this method.")

        dataset = Dataset(table_name)
        return dataset.schema()

    def get_partitions(self, table_name: str) -> List[str]:
        """
        Retrieve the list of partitions for a table.

        Args:
            table_name (str): The name of the table to query.

        Returns:
            List[str]: A list of partition names.
        """
        try:
            from deltacat import Dataset
        except ImportError:
            raise ImportError("deltacat must be installed to use this method.")

        dataset = Dataset(table_name)
        return dataset.partitions()

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Retrieve metadata information about a table.

        Args:
            table_name (str): The name of the table to query.

        Returns:
            Dict[str, Any]: A dictionary containing table metadata.
        """
        try:
            from deltacat import Dataset
        except ImportError:
            raise ImportError("deltacat must be installed to use this method.")

        dataset = Dataset(table_name)
        return {
            "name": table_name,
            "schema": self.get_schema(table_name),
            "partitions": self.get_partitions(table_name),
            "num_rows": dataset.num_rows(),
            "num_columns": len(dataset.schema().names),
            "created_at": dataset.created_at(),
            "last_modified": dataset.last_modified(),
        }


class DataSelector:
    def __init__(self):
        self.selected_data = None

    def select(
        self,
        data_object: DataObject,
        columns: List[str] = None,
        filters: Dict[str, Any] = None,
    ):
        """
        Select data from a DataObject with optional column selection and filtering.

        Args:
            data_object (DataObject): The DataObject to select from.
            columns (List[str], optional): List of columns to select. If None, select all columns.
            filters (Dict[str, Any], optional): Dictionary of filters to apply. Keys are column names, values are filter conditions.

        Returns:
            DataSelector: The DataSelector instance for method chaining.
        """
        try:
            import daft
        except ImportError:
            raise ImportError("daft must be installed to use this method.")

        df = daft.from_pyarrow_table(data_object.to_arrow())

        if columns:
            df = df.select(columns)

        if filters:
            for column, condition in filters.items():
                df = df.filter(daft.col(column) == condition)

        self.selected_data = df
        return self

    def join(self, other: "DataSelector", on: str, how: str = "inner"):
        """
        Join the current selected data with another DataSelector's data.

        Args:
            other (DataSelector): Another DataSelector instance to join with.
            on (str): The column name to join on.
            how (str): The type of join to perform ('inner', 'left', 'right', 'outer').

        Returns:
            DataSelector: The DataSelector instance for method chaining.
        """
        if self.selected_data is None or other.selected_data is None:
            raise ValueError(
                "Both DataSelectors must have selected data before joining."
            )

        self.selected_data = self.selected_data.join(
            other.selected_data, on=on, how=how
        )
        return self

    def to_daft(self) -> "daft.DataFrame":
        """
        Convert the selected and processed data to a Daft DataFrame.

        Returns:
            daft.DataFrame: The resulting Daft DataFrame.
        """
        if self.selected_data is None:
            raise ValueError(
                "No data has been selected. Use the 'select' method first."
            )

        return self.selected_data

    def to_arrow(self) -> pa.Table:
        """
        Convert the selected and processed data to a PyArrow Table.

        Returns:
            pa.Table: The resulting PyArrow Table.
        """
        if self.selected_data is None:
            raise ValueError(
                "No data has been selected. Use the 'select' method first."
            )

        return self.selected_data.to_arrow()
