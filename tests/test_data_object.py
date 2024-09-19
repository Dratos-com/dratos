import unittest
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import daft
import pyarrow as pa
from datetime import datetime, timezone
from beta.data.obj import DataObject


class TestDataObject(unittest.TestCase):
    def setUp(self):
        self.data_object = DataObject()

    def test_initialization(self):
        self.assertIsInstance(self.data_object, DataObject)
        self.assertIsInstance(self.data_object.df, daft.DataFrame)
        self.assertEqual(len(self.data_object.df), 0)

    def test_schema(self):
        expected_columns = ["id", "created_at", "updated_at", "inserted_at", "type"]
        self.assertEqual(set(self.data_object.df.schema.names), set(expected_columns))

    def test_add_obj_dict(self):
        test_data = {
            "id": "01234567-89AB-CDEF-GHIJ-KLMNOPQRSTUV",
            "type": "TestType",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "inserted_at": datetime.now(timezone.utc),
        }
        self.data_object.add_obj(test_data)
        self.assertEqual(len(self.data_object.df), 1)
        self.assertEqual(self.data_object.df["type"][0], "TestType")

    def test_add_obj_daft_dataframe(self):
        test_data = daft.from_pydict(
            {
                "id": ["01234567-89AB-CDEF-GHIJ-KLMNOPQRSTUV"],
                "type": ["TestType"],
                "created_at": [datetime.now(timezone.utc)],
                "updated_at": [datetime.now(timezone.utc)],
                "inserted_at": [datetime.now(timezone.utc)],
            }
        )
        self.data_object.add_obj(test_data)
        self.assertEqual(len(self.data_object.df), 1)
        self.assertEqual(self.data_object.df["type"][0], "TestType")

    def test_add_obj_pyarrow_table(self):
        test_data = pa.table(
            {
                "id": ["01234567-89AB-CDEF-GHIJ-KLMNOPQRSTUV"],
                "type": ["TestType"],
                "created_at": [datetime.now(timezone.utc)],
                "updated_at": [datetime.now(timezone.utc)],
                "inserted_at": [datetime.now(timezone.utc)],
            }
        )
        self.data_object.add_obj(test_data)
        self.assertEqual(len(self.data_object.df), 1)
        self.assertEqual(self.data_object.df["type"][0], "TestType")

    def test_add_obj_missing_type(self):
        test_data = {
            "id": "01234567-89AB-CDEF-GHIJ-KLMNOPQRSTUV",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "inserted_at": datetime.now(timezone.utc),
        }
        self.data_object.add_obj(test_data)
        self.assertEqual(len(self.data_object.df), 1)
        self.assertEqual(self.data_object.df["type"][0], "DataObject")

    def test_concat(self):
        test_data1 = {"id": ["1"], "type": ["Type1"]}
        test_data2 = {"id": ["2"], "type": ["Type2"]}
        self.data_object.add_obj(test_data1)
        self.data_object.concat(test_data2)
        self.assertEqual(len(self.data_object.df), 2)
        self.assertEqual(self.data_object.df["type"][1], "Type2")

    def test_to_arrow(self):
        test_data = {"id": ["1"], "type": ["TestType"]}
        self.data_object.add_obj(test_data)
        arrow_table = self.data_object.to_arrow()
        self.assertIsInstance(arrow_table, pa.Table)
        self.assertEqual(arrow_table["type"][0].as_py(), "TestType")

    def test_to_pydict(self):
        test_data = {"id": ["1"], "type": ["TestType"]}
        self.data_object.add_obj(test_data)
        py_dict = self.data_object.to_pydict()
        self.assertIsInstance(py_dict, dict)
        self.assertEqual(py_dict["type"][0], "TestType")

    def test_repr(self):
        test_data = {"id": ["1"], "type": ["TestType"]}
        self.data_object.add_obj(test_data)
        self.assertEqual(repr(self.data_object), "<DataObject with 1 entries>")

    def test_show(self):
        test_data = {"id": ["1"], "type": ["TestType"]}
        self.data_object.add_obj(test_data)
        # This is a bit tricky to test as show() prints to stdout
        # We could potentially capture stdout and check its content
        # For now, we'll just ensure it doesn't raise an exception
        try:
            self.data_object.show()
        except Exception as e:
            self.fail(f"show() raised {type(e).__name__} unexpectedly!")

    def test_multiple_additions(self):
        for i in range(5):
            self.data_object.add_obj({"id": [str(i)], "type": [f"Type{i}"]})
        self.assertEqual(len(self.data_object.df), 5)
        self.assertEqual(self.data_object.df["type"][4], "Type4")

    def test_add_obj_with_extra_columns(self):
        test_data = {"id": ["1"], "type": ["TestType"], "extra_column": ["ExtraData"]}
        self.data_object.add_obj(test_data)
        self.assertEqual(len(self.data_object.df), 1)
        self.assertNotIn("extra_column", self.data_object.df.schema.names)


if __name__ == "__main__":
    unittest.main()
