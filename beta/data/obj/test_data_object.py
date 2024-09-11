class TestDataObject(unittest.TestCase):
    def setUp(self):
        class TestObject(DataObject):
            string_field: str
            int_field: int
            float_field: float
            bool_field: bool
            datetime_field: datetime
            date_field: date
            time_field: time
            uuid_field: uuid.UUID
            ulid_field: str
            url_field: HttpUrl
            email_field: EmailStr
            list_field: List[str]
            dict_field: Dict[str, Any]

        self.TestObject = TestObject
        self.test_data = {
            "string_field": "test",
            "int_field": 42,
            "float_field": 3.14,
            "bool_field": True,
            "datetime_field": datetime.now(),
            "date_field": date.today(),
            "time_field": datetime.now().time(),
            "uuid_field": uuid.uuid4(),
            "ulid_field": str(ulid.new()),
            "url_field": "https://example.com",
            "email_field": "test@example.com",
            "list_field": ["a", "b", "c"],
            "dict_field": {"key": "value"},
        }

    def test_create_data_object(self):
        obj = self.TestObject(**self.test_data)
        self.assertIsInstance(obj, DataObject)
        self.assertEqual(obj.string_field, self.test_data["string_field"])
        self.assertEqual(obj.int_field, self.test_data["int_field"])
        self.assertEqual(obj.float_field, self.test_data["float_field"])
        self.assertEqual(obj.bool_field, self.test_data["bool_field"])
        self.assertEqual(obj.datetime_field, self.test_data["datetime_field"])
        self.assertEqual(obj.date_field, self.test_data["date_field"])
        self.assertEqual(obj.time_field, self.test_data["time_field"])
        self.assertEqual(obj.uuid_field, self.test_data["uuid_field"])
        self.assertEqual(obj.ulid_field, self.test_data["ulid_field"])
        self.assertEqual(str(obj.url_field), self.test_data["url_field"])
        self.assertEqual(str(obj.email_field), self.test_data["email_field"])
        self.assertEqual(obj.list_field, self.test_data["list_field"])
        self.assertEqual(obj.dict_field, self.test_data["dict_field"])

    def test_to_arrow(self):
        obj = self.TestObject(**self.test_data)
        arrow_table = obj.to_arrow()
        self.assertIsInstance(arrow_table, pa.Table)
        self.assertEqual(len(arrow_table), 1)
        self.assertEqual(
            len(arrow_table.schema), len(self.test_data) + 3
        )  # +3 for id, updated_at, and metadata

    def test_from_arrow(self):
        obj = self.TestObject(**self.test_data)
        arrow_table = obj.to_arrow()
        new_obj = self.TestObject.from_arrow(arrow_table)
        self.assertEqual(obj.dict(), new_obj.dict())

    def test_to_json_schema(self):
        schema = self.TestObject.to_json_schema()
        self.assertIsInstance(schema, dict)
        self.assertIn("properties", schema)
        self.assertIn("string_field", schema["properties"])
        self.assertIn("int_field", schema["properties"])
        self.assertIn("float_field", schema["properties"])

    def test_from_json(self):
        obj = self.TestObject(**self.test_data)
        json_data = obj.to_json()
        new_obj = self.TestObject.from_json(json_data)
        self.assertEqual(obj.dict(), new_obj.dict())

    def test_to_json(self):
        obj = self.TestObject(**self.test_data)
        json_data = obj.to_json()
        self.assertIsInstance(json_data, str)
        parsed_data = json.loads(json_data)
        self.assertIn("string_field", parsed_data)
        self.assertIn("int_field", parsed_data)
        self.assertIn("float_field", parsed_data)

    def test_get_arrow_schema(self):
        schema = self.TestObject.get_arrow_schema()
        self.assertIsInstance(schema, pa.Schema)
        self.assertIn("string_field", schema.names)
        self.assertIn("int_field", schema.names)
        self.assertIn("float_field", schema.names)

    def test_to_arrow_batch(self):
        obj1 = self.TestObject(**self.test_data)
        obj2 = self.TestObject(**self.test_data)
        batch = self.TestObject.to_arrow_batch([obj1, obj2])
        self.assertIsInstance(batch, pa.RecordBatch)
        self.assertEqual(len(batch), 2)

    def test_from_arrow_batch(self):
        obj1 = self.TestObject(**self.test_data)
        obj2 = self.TestObject(**self.test_data)
        batch = self.TestObject.to_arrow_batch([obj1, obj2])
        objects = self.TestObject.from_arrow_batch(batch)
        self.assertEqual(len(objects), 2)
        self.assertEqual(objects[0].dict(), obj1.dict())
        self.assertEqual(objects[1].dict(), obj2.dict())

    def test_to_arrow_table(self):
        obj1 = self.TestObject(**self.test_data)
        obj2 = self.TestObject(**self.test_data)
        table = self.TestObject.to_arrow_table([obj1, obj2])
        self.assertIsInstance(table, pa.Table)
        self.assertEqual(len(table), 2)

    def test_from_arrow_table(self):
        obj1 = self.TestObject(**self.test_data)
        obj2 = self.TestObject(**self.test_data)
        table = self.TestObject.to_arrow_table([obj1, obj2])
        objects = self.TestObject.from_arrow_table(table)
        self.assertEqual(len(objects), 2)
        self.assertEqual(objects[0].dict(), obj1.dict())
        self.assertEqual(objects[1].dict(), obj2.dict())

    def test_to_dataframe(self):
        obj1 = self.TestObject(**self.test_data)
        obj2 = self.TestObject(**self.test_data)
        df = self.TestObject.to_dataframe([obj1, obj2])
        self.assertEqual(len(df), 2)

    def test_invalid_arrow_conversion(self):
        invalid_data = {"invalid_field": "invalid_value"}
        with self.assertRaises(ArrowConversionError):
            self.TestObject(**invalid_data).to_arrow()
