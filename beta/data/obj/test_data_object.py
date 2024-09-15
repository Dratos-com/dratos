import unittest
import datetime
import pyarrow as pa
from ulid import ULID

from data_object import (
    DataObject,
    ArrowConversionError,
)
import daft


class TestDataObject(unittest.TestCase):
    def test_default_id(self):
        """Test that the default ID is a valid ULID."""
        obj = DataObject()
        self.assertIsNotNone(obj.id)
        self.assertIsInstance(obj.id, ULID)
        self.assertEqual(len(str(obj.id)), 26)

    def test_created_at(self):
        """Test the created_at property for correct timestamp extraction."""
        obj = DataObject()
        self.assertIsInstance(obj.created_at, datetime.datetime)
        self.assertEqual(obj.created_at, obj.updated_at)
        self.assertEqual(obj.created_at.tzinfo, datetime.timezone.utc)

    def test_update_timestamp(self):
        """Test the update_timestamp method updates 'updated_at' correctly."""
        obj = DataObject()
        original_updated_at = obj.updated_at
        obj.update_timestamp()
        self.assertIsNotNone(obj.updated_at)
        self.assertNotEqual(obj.updated_at, original_updated_at)
        self.assertNotEqual(obj.created_at, obj.updated_at)

    def test_get_schema(self):
        """Test that get_schema returns a valid schema dictionary."""
        schema = DataObject.get_schema()
        self.assertIsInstance(schema, dict)
        self.assertIn("id", schema["properties"])
        self.assertIn("created_at", schema["properties"])
        self.assertIn("updated_at", schema["properties"])
        self.assertIn("metadata", schema["properties"])

    def test_get_arrow_schema(self):
        """Test that get_arrow_schema returns a valid PyArrow schema."""
        schema = DataObject.get_arrow_schema()
        self.assertIsInstance(schema, pa.Schema)
        self.assertIn("id", schema.names)
        self.assertIn("updated_at", schema.names)
        self.assertIn("metadata", schema.names)

    def test_get_daft_schema(self):
        """Test that get_daft_schema returns a valid Daft schema."""
        schema = DataObject.get_daft_schema()
        self.assertIsInstance(schema, daft.Schema)
        self.assertIn("id", schema.column_names())
        self.assertIn("updated_at", schema.column_names())
        self.assertIn("metadata", schema.column_names())

    def test_arrow_serialization(self):
        """Test serialization to and from Arrow format."""
        obj = DataObject(metadata={"key": "value"})
        obj.update_timestamp()

        # Serialize to Arrow
        arrow_table = pa.Table.from_pylist(
            [obj.model_dump()], schema=DataObject.get_arrow_schema()
        )

        # Deserialize from Arrow
        deserialized_obj = DataObject.model_validate(arrow_table.to_pylist()[0])

        self.assertEqual(obj, deserialized_obj)

    def test_exception_handling(self):
        """Test custom exceptions for expected failure scenarios."""
        with self.assertRaises(ArrowConversionError):
            # Simulate a scenario where Arrow conversion fails
            DataObject.get_arrow_schema = lambda: None
            DataObject.get_arrow_schema()

        # Reset the get_arrow_schema method
        DataObject.get_arrow_schema = lambda: pa.schema(
            [
                ("id", pa.string()),
                ("updated_at", pa.timestamp("us", tz="UTC")),
                ("metadata", pa.map_(pa.string(), pa.string())),
            ]
        )


if __name__ == "__main__":
    unittest.main()
