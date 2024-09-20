from pydantic import BaseModel, ValidationError
from typing import Dict, TypeVar, Generic
from .result import Result

T = TypeVar('T')

class MetadataSchema(BaseModel):
    schema: str
    timestamp: datetime
    provenance: Optional[str]

class SchemaValidator:
    def __init__(self, schemas: Dict[str, Type[BaseModel]]):
        self.schemas = schemas

    def validate(self, obj: T, schema_name: str) -> Result[T, ValidationError]:
        schema = self.schemas.get(schema_name)
        if not schema:
            return Result.Error(ValueError(f"Schema {schema_name} not found."))
        try:
            validated_obj = schema(**obj.__dict__)
            print(f"Validation successful for schema: {schema_name}")
            return Result.Ok(validated_obj)
        except ValidationError as e:
            return Result.Error(e)