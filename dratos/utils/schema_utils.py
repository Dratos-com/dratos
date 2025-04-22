from enum import Enum
from pydantic import BaseModel, ValidationError, Field
from typing import Dict, Any, Optional, List, Type, Union
from pydantic.main import create_model


class SchemaType(str, Enum):
    """Enum for schema data types."""
    STRING = 'string'
    NUMBER = 'number'
    INTEGER = 'integer'
    BOOLEAN = 'boolean'
    ARRAY = 'array'
    OBJECT = 'object'


class Schema(BaseModel):
    """
    Schema that defines the format of input and output data.
    Represents a select subset of an OpenAPI 3.0 schema object.
    """
    min_items: Optional[int] = None
    example: Optional[Any] = None
    property_ordering: Optional[List[str]] = None
    pattern: Optional[str] = None
    minimum: Optional[float] = None
    default: Optional[Any] = None
    any_of: Optional[List['Schema']] = None
    max_length: Optional[int] = None
    title: Optional[str] = None
    min_length: Optional[int] = None
    min_properties: Optional[int] = None
    max_items: Optional[int] = None
    maximum: Optional[float] = None
    nullable: Optional[bool] = None
    max_properties: Optional[int] = None
    type: Optional[SchemaType] = None
    description: Optional[str] = None
    enum: Optional[List[str]] = None
    format: Optional[str] = None
    items: Optional['Schema'] = None
    properties: Optional[Dict[str, 'Schema']] = None
    required: Optional[List[str]] = None

Schema.model_rebuild() # Rebuilds the model's internal schema after any forward references

def validate_schema(schema: dict, path="root") -> bool:
    """
    Recursively validates a JSON Schema structure.
    Ensures that it follows OpenAPI 3.0 schema constraints.
    """
    try:
        Schema.model_validate(schema)  # Validate using Pydantic
    except ValidationError as e:
        raise ValueError(f"Schema validation error at '{path}': {e}")

    if schema.get("type") != "object":
        raise ValueError(f"Root schema must have 'type': 'object' at '{path}'")

    if "properties" not in schema or not isinstance(schema["properties"], dict):
        raise ValueError(f"Schema at '{path}' must contain a 'properties' field of type object")

    properties = schema["properties"]
    
    def validate_object(properties: dict, path: str, required_fields: list = None):
        """Validates nested objects and arrays recursively."""
        # Check that required fields exist in properties
        if required_fields:
            if not all(isinstance(field, str) for field in required_fields):
                raise ValueError(f"'required' at '{path}' must be a list of field names")
            
            missing_fields = [field for field in required_fields if field not in properties]
            if missing_fields:
                raise ValueError(f"Required fields {missing_fields} at '{path}' are missing in 'properties'")
        
        for prop_name, definition in properties.items():
            prop_path = f"{path}.{prop_name}"

            if not isinstance(definition, dict):
                raise ValueError(f"Property '{prop_path}' must be an object")

            prop_type = definition.get("type")
            if not prop_type:
                raise ValueError(f"Property '{prop_path}' must have a 'type'")

            if prop_type == "array":
                if "items" not in definition:
                    raise ValueError(f"Array property '{prop_path}' must specify 'items'")
                validate_object({"items": definition["items"]}, prop_path)  # Validate array schema

            elif prop_type == "object":
                if "properties" not in definition or not isinstance(definition["properties"], dict):
                    raise ValueError(f"Object property '{prop_path}' must define 'properties'")
                # Validate nested object's required fields against its properties
                validate_object(
                    definition["properties"], 
                    prop_path,
                    definition.get("required", [])
                )

    # Validate all properties and check required fields at root level
    validate_object(properties, path, schema.get("required", []))

    return True  # Schema is valid

def create_model_from_schema(schema: dict) -> Type[BaseModel]:
    """
    Creates a Pydantic BaseModel class from a JSON schema.
    
    Args:
        schema: A JSON schema following OpenAPI 3.0 format
        
    Returns:
        A dynamically created Pydantic BaseModel class
        
    Raises:
        ValueError: If the schema contains unsupported fields or structures
    """
    # Validate the schema first
    validate_schema(schema)
    
    # Track unsupported fields that are used in the schema
    unsupported_fields = set()
    supported_fields = {
        'type', 'title', 'items', 'properties', 'required',
        'description', 'enum', 'default', 'minimum', 'maximum',
        'min_length', 'max_length', 'pattern', 'min_items', 'max_items',
        'nullable', 'format', 'any_of'
    }
    
    # Model cache to handle nested models
    model_cache = {}
    
    def process_schema(schema_def, is_required=True):
        """Process a schema definition and return the appropriate Python type and field constraints"""
        # Check for unsupported fields
        for field in schema_def:
            if field not in supported_fields:
                unsupported_fields.add(field)
        
        # Handle any_of
        if "any_of" in schema_def:
            union_types = [process_schema(subschema, is_required)[0] for subschema in schema_def["any_of"]]
            return Union[tuple(union_types)], {}
        
        schema_type = schema_def.get("type")
        field_args = {}
        
        # Add description if available
        if "description" in schema_def:
            field_args["description"] = schema_def["description"]
            
        # Add default if available
        if "default" in schema_def:
            field_args["default"] = schema_def["default"]
        
        # Handle nullable fields
        nullable = schema_def.get("nullable", False)
        
        if schema_type == SchemaType.STRING.value:
            field_type = str
            
            # Add string-specific constraints
            if "min_length" in schema_def:
                field_args["min_length"] = schema_def["min_length"]
            if "max_length" in schema_def:
                field_args["max_length"] = schema_def["max_length"]
            if "pattern" in schema_def:
                field_args["pattern"] = schema_def["pattern"]
            if "enum" in schema_def:
                field_args["enum"] = schema_def["enum"]
                
            # Handle format (like date-time, email, etc.)
            if "format" in schema_def:
                # You could use specialized types based on format
                # For simplicity, just adding as a field constraint
                field_args["format"] = schema_def["format"]
                
        elif schema_type == SchemaType.NUMBER.value:
            field_type = float
            
            # Add number-specific constraints
            if "minimum" in schema_def:
                field_args["ge"] = schema_def["minimum"]
            if "maximum" in schema_def:
                field_args["le"] = schema_def["maximum"]
                
        elif schema_type == SchemaType.INTEGER.value:
            field_type = int
            
            # Add integer-specific constraints
            if "minimum" in schema_def:
                field_args["ge"] = schema_def["minimum"]
            if "maximum" in schema_def:
                field_args["le"] = schema_def["maximum"]
                
        elif schema_type == SchemaType.BOOLEAN.value:
            field_type = bool
            
        elif schema_type == SchemaType.ARRAY.value:
            item_type, _ = process_schema(schema_def.get("items", {}))
            field_type = List[item_type]
            
            # Add array-specific constraints
            if "min_items" in schema_def:
                field_args["min_items"] = schema_def["min_items"]
            if "max_items" in schema_def:
                field_args["max_items"] = schema_def["max_items"]
                
        elif schema_type == SchemaType.OBJECT.value:
            # Create a nested model for object types
            nested_name = schema_def.get("title", f"NestedModel{len(model_cache)}")
            
            if nested_name in model_cache:
                return model_cache[nested_name], {}
                
            nested_properties = schema_def.get("properties", {})
            nested_required = schema_def.get("required", [])
            nested_fields = {}
            
            # Create placeholder to avoid circular references
            placeholder_model = type(nested_name, (BaseModel,), {})
            model_cache[nested_name] = placeholder_model
            
            for prop_name, prop_schema in nested_properties.items():
                prop_required = prop_name in nested_required
                prop_type, prop_field_args = process_schema(prop_schema, prop_required)
                
                if not prop_required:
                    if prop_field_args and "default" in prop_field_args:
                        nested_fields[prop_name] = (prop_type, Field(**prop_field_args))
                    else:
                        nested_fields[prop_name] = (Optional[prop_type], Field(default=None, **prop_field_args))
                else:
                    nested_fields[prop_name] = (prop_type, Field(..., **prop_field_args))
            
            # Create the actual model and update cache
            nested_model = create_model(nested_name, **nested_fields)
            model_cache[nested_name] = nested_model
            field_type = nested_model
            
        else:
            field_type = Any
        
        # Handle nullable fields
        if nullable and schema_type != SchemaType.OBJECT.value:
            field_type = Optional[field_type]
            if "default" not in field_args:
                field_args["default"] = None
        
        return field_type, field_args
    
    # Check if there are unsupported fields being used
    if unsupported_fields:
        raise ValueError(f"Schema contains unsupported fields: {', '.join(unsupported_fields)}")
    
    # Process the root schema
    model_name = schema.get("title", "Schema")
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])
    field_definitions = {}
    
    for field_name, field_schema in properties.items():
        field_required = field_name in required_fields
        field_type, field_args = process_schema(field_schema, field_required)
        
        if field_required:
            field_definitions[field_name] = (field_type, Field(..., **field_args))
        else:
            if "default" in field_args:
                field_definitions[field_name] = (field_type, Field(**field_args))
            else:
                field_definitions[field_name] = (Optional[field_type], Field(default=None, **field_args))
    
    # Create the model
    return create_model(model_name, **field_definitions)
