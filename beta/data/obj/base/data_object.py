from typing import Optional, Dict, Any, Union, ClassVar, List
from ulid import ULID
import pyarrow as pa
import daft
from daft import col
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_object_schema = pa.schema(
    [
        pa.field(
            "id",
            pa.string(),
            nullable=False,
            metadata={"description": "ULID Unique identifier for the record"},
        ),
        pa.field(
            "type",
            pa.string(),
            nullable=False,
            metadata={"description": "Type of the data object"},
        ),
        pa.field(
            "created_at",
            pa.timestamp("ns", tz="UTC"),
            nullable=False,
            metadata={"description": "Timestamp when the record was created"},
        ),
        pa.field(
            "updated_at",
            pa.timestamp("ns", tz="UTC"),
            nullable=False,
            metadata={"description": "Timestamp when the record was last updated"},
        ),
        pa.field(
            "inserted_at",
            pa.timestamp("ns", tz="UTC"),
            nullable=False,
            metadata={
                "description": "Timestamp when the data object was inserted into the database"
            },
        ),
    ]
)


class DataObject:
    """
    Base class for all data objects in the system.
    Provides common fields and methods for Arrow conversion and data manipulation.

    dob = DataObject(data = [
        {"id": str(ULID()), "type": "test", "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc), "inserted_at": datetime.now(timezone.utc)}])
    """

    schema: ClassVar[pa.Schema] = data_object_schema
    obj_type: ClassVar[str] = "DataObject"

    def __init__(
        self,
        data: Optional[
            Union[daft.DataFrame, pa.Table, Dict[str, Any], "DataObject"]
        ] = None,
    ):
        if data is not None:
            df = self.to_daft(data)
            df = df.select(*self.schema.names)
            self.df = df
        else:
            # Initialize an empty DataFrame with the defined schema
            empty_data = {field.name: [] for field in self.schema}
            self.df = daft.from_pydict(empty_data)

    @staticmethod
    def to_daft(
        data: Union[daft.DataFrame, pa.Table, Dict[str, Any], "DataObject"],
    ) -> daft.DataFrame:
        """
        Convert the provided data to a Daft DataFrame.

        Args:
            data: The data to convert.

        Returns:
            A Daft DataFrame.

        Raises:
            ValueError: If the data type is invalid.
        """
        if isinstance(data, daft.DataFrame):
            return data
        elif isinstance(data, pa.Table):
            return daft.from_arrow(data)
        elif isinstance(data, Dict):
            return daft.from_pydict(data)
        elif isinstance(data, DataObject):
            return data.df
        else:
            raise ValueError(
                f"Invalid data type: {type(data)}. "
                "Expected daft.DataFrame, pa.Table, Dict[str, Any], or DataObject."
            )

    def concat(
        self, data: Union[daft.DataFrame, pa.Table, Dict[str, Any]]
    ) -> "DataObject":
        """
        Concatenate the provided data to the existing data.

        Args:
            data: The data to concatenate.

        Returns:
            The DataObject instance (self).
        """
        new_data = self.to_daft(data)
        new_data = new_data.select(*self.schema.names)
        self.df = self.df.concat(new_data)
        return self

    def validate(self) -> None:
        """
        Validate the data object by checking schema, IDs, and timestamps.

        Raises:
            ValueError: If validation fails.
        """
        logger.info("Validating DataObject...")
        self.validate_id()
        self.validate_timestamps()
        self.validate_schema()
        logger.info("Validation successful.")

    def validate_schema(self) -> None:
        """
        Validate the schema of the data object.

        Raises:
            ValueError: If the DataFrame schema does not match the expected schema.
        """
        if self.df.schema != self.schema:
            raise ValueError(
                f"DataFrame schema does not match the expected schema.\n"
                f"Expected: {self.schema}\nActual: {self.df.schema}"
            )

    def validate_id(self) -> None:
        """
        Validate the 'id' field to ensure all IDs are valid ULIDs.

        Raises:
            ValueError: If any IDs are invalid.
        """
        logger.info("Validating 'id' field for valid ULIDs...")
        is_valid_ulid = self.df["id"].apply(
            lambda x: ULID.is_valid(x), return_type=pa.bool_()
        )
        invalid_ids_df = self.df.filter(~is_valid_ulid).select("id")
        invalid_ids = invalid_ids_df.to_pydict().get("id", [])
        if invalid_ids:
            raise ValueError(f"Invalid ULIDs found: {invalid_ids}")
        logger.info("'id' field validation passed.")

    def validate_timestamps(
        self, columns: List[str] = ["created_at", "updated_at", "inserted_at"]
    ) -> None:
        """
        Validate timestamp fields to ensure they are timezone-aware and in UTC.

        Args:
            columns: List of columns to validate.

        Raises:
            ValueError: If any timestamps are invalid.
        """
        logger.info("Validating timestamp fields...")
        for column in columns:
            if column not in self.df.column_names():
                raise ValueError(f"Column '{column}' not found in DataFrame.")
            column_schema = self.df.schema.get_field(column).data_type
            if not pa.types.is_timestamp(column_schema):
                raise TypeError(f"Column '{column}' must be of timestamp type.")
            # Check timezone
            timestamp_type = column_schema
            if timestamp_type.tz != "UTC":
                raise ValueError(f"Timestamp column '{column}' must be in UTC.")
        logger.info("Timestamp fields validation passed.")

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another DataObject.

        Args:
            other: The other object to compare.

        Returns:
            True if equal, False otherwise.
        """
        if not isinstance(other, DataObject):
            return NotImplemented
        return self.df.collect() == other.df.collect()

    def __ne__(self, other: object) -> bool:
        """
        Check inequality with another DataObject.

        Args:
            other: The other object to compare.

        Returns:
            True if not equal, False otherwise.
        """
        return not self.__eq__(other)

    def __repr__(self) -> str:
        """
        String representation of the DataObject.

        Returns:
            A string representation.
        """
        count = self.df.count()
        return f"<{self.__class__.__name__} with {count} entries>"

    def __str__(self) -> str:
        """
        String representation (same as __repr__).

        Returns:
            A string representation.
        """
        return self.__repr__()


if __name__ == "__main__":
    import unittest

    class TestDataObject(unittest.TestCase):
        def setUp(self):
            self.valid_data = {
                "id": [str(ULID())],
                "type": ["TestType"],
                "created_at": [datetime.now(timezone.utc)],
                "updated_at": [datetime.now(timezone.utc)],
                "inserted_at": [datetime.now(timezone.utc)],
            }
            self.invalid_id_data = {
                "id": ["InvalidULID"],
                "type": ["TestType"],
                "created_at": [datetime.now(timezone.utc)],
                "updated_at": [datetime.now(timezone.utc)],
                "inserted_at": [datetime.now(timezone.utc)],
            }
            self.invalid_timestamp_data = {
                "id": [str(ULID())],
                "type": ["TestType"],
                "created_at": [datetime.now()],  # No timezone
                "updated_at": [datetime.now()],
                "inserted_at": [datetime.now()],
            }

        def test_init_with_valid_data(self):
            obj = DataObject(self.valid_data)
            self.assertEqual(obj.df.count(), 1)
            self.assertEqual(obj.df.schema, data_object_schema)

        def test_validate_with_valid_data(self):
            obj = DataObject(self.valid_data)
            obj.validate()  # Should not raise an exception

        def test_validate_invalid_id(self):
            obj = DataObject(self.invalid_id_data)
            with self.assertRaises(ValueError) as context:
                obj.validate()
            self.assertIn("Invalid ULIDs found", str(context.exception))

        def test_validate_invalid_timestamps(self):
            obj = DataObject(self.invalid_timestamp_data)
            with self.assertRaises(ValueError) as context:
                obj.validate()
            self.assertIn("must be timezone-aware and in UTC", str(context.exception))

        def test_concat(self):
            obj1 = DataObject(self.valid_data)
            obj2 = DataObject(self.valid_data)
            obj1.concat(obj2)
            self.assertEqual(obj1.df.count(), 2)

        def test_equality(self):
            obj1 = DataObject(self.valid_data)
            obj2 = DataObject(self.valid_data)
            self.assertEqual(obj1, obj2)
            obj2.concat(self.valid_data)
            self.assertNotEqual(obj1, obj2)

    unittest.main()
