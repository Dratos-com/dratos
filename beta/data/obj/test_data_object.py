import unittest
import datetime
import pyarrow as pa
from ulid import ULID

from data_object import (
    DataObject,
    ULIDValidationError,
    ArrowConversionError,
)


class TestDataObject(unittest.TestCase):
    def test_default_id(self):
        """Test that the default ID is a valid ULID."""
        obj = DataObject()
        self.assertIsNotNone(obj.id)  # Ensure ID is not None

        # Check that the ID is a valid ULID
        self.assertIsInstance(obj.id, ULID)
        self.assertEqual(len(obj.id), 26)

    def test_created_at(self):
        """Test the created_at property for correct timestamp extraction."""
        obj = DataObject()

        # Check that obj.created_at is a datetime object
        self.assertIsInstance(obj.created_at, datetime.datetime)

        # Check that created_at and updated_at are the same
        self.assertEqual(obj.created_at, obj.updated_at)

        # Check that timestamp is in UTC
        self.assertEqual(obj.created_at.tzinfo, datetime.timezone.utc)

    def test_update_timestamp(self):
        """Test the update_timestamp method updates 'updated_at' correctly."""
        obj = DataObject()
        obj.update_timestamp()
        self.assertIsNotNone(obj.updated_at)
        self.assertAlmostEqual(
            obj.updated_at,
            datetime.now(datetime.timezone.utc),
            delta=datetime.timedelta(seconds=1),
        )

    def test_get_schema(self):
        """Test that get_schema returns a valid schema dictionary."""
        schema = DataObject.get_schema()
        self.assertIsInstance(schema, dict)

    def test_get_arrow_schema(self):
        """Test that get_arrow_schema returns a valid PyArrow schema."""
        schema = DataObject.get_arrow_schema()
        self.assertIsInstance(schema, pa.Schema)

    def test_arrow_serialization(self):
        """Test serialization to and from Arrow format."""
        obj = DataObject(
            metadata={"key": "value"}, updated_at=None
        )  # Set updated_at to None for testing

        # Serialize to Arrow
        arrow_table = pa.Table.from_pylist(
            [obj.dict()], schema=DataObject.get_arrow_schema()
        )

        # Deserialize from Arrow
        deserialized_obj = DataObject.parse_obj(arrow_table.to_pylist()[0])

        self.assertEqual(obj, deserialized_obj)

    def test_exception_handling(self):
        """Test custom exceptions for expected failure scenarios."""
        with self.assertRaises(ULIDValidationError):
            # Assuming there's a method to validate ULID that can raise ULIDValidationError
            DataObject.validate_ulid("invalid_ulid")

        with self.assertRaises(ArrowConversionError):
            # Assuming there's a method that could raise ArrowConversionError
            DataObject.convert_to_arrow("invalid_data")


if __name__ == "__main__":
    unittest.main()
