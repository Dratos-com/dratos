class PydanticObject(BaseModel):
    id: str
    uri: str
    name: str
    description: str
    tags: List[str]
    metadata: Dict[str, Any]

obj = PydanticObject(id="123", uri="s3://bucket/key", name="name", description="description", tags=["tag1", "tag2"], metadata={"key": "value"})
df = daft.from_pydict(obj.model_dump())
big_df = daft.from_pylist([obj.model_dump(), obj.model_dump(), obj.model_dump()])
big_df.write_lance(path="s3://bucket/lance/table")