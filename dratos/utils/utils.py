import json
import inspect
from pydantic import BaseModel
from typing import Callable, Dict

def extract_json_from_str(response: str):
    """
    Extracts a JSON object from a string.

    Args:
        response (str): The string containing the JSON object.

    Returns:
        dict: The extracted JSON object.

    Raises:
        ValueError: If the string does not contain a valid JSON object.
    """
    try:
        json_start = response.index("{")
        json_end = response.rfind("}")
        return json.loads(response[json_start : json_end + 1]), response[0:json_start], response[json_end + 1:]
    except Exception as e:
        raise ValueError("No valid JSON structure found in the input string")

def function_to_openai_definition(tool: Callable) -> Dict:
    name = tool.__name__
    desc = inspect.getdoc(tool) or f"{name} definition."
    params = inspect.signature(tool).parameters

    type_map = {
        "int": "integer",
        "str": "string",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "tuple": "array",
        "set": "array",
        "NoneType": "null",
        "bytes": "string",
        "complex": "number",
    }

    def get_type(ann):
        py_type = ann.__name__.lower() if ann != inspect.Parameter.empty else "string"
        if py_type not in type_map:
            raise ValueError(f"Type '{py_type}' not found. Add it to 'type_map' in 'utils' under 'core'.")
        return type_map[py_type]

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    p: {
                        "type": get_type(param.annotation),
                        "description": f"{p} parameter."
                    } for p, param in params.items()
                },
                "required": list(params.keys()),
                "additionalProperties": False
            }
        }
    }

def _pydantic_to_openai_schema(schema: dict) -> dict:
    """Convert Pydantic schema to OpenAI schema."""
    openai_schema = {
        "properties": schema.get("properties", {}),
    }
    if "required" in schema:
        openai_schema["required"] = schema["required"]
    if "description" in schema:
        openai_schema["description"] = schema["description"]
    if schema["type"] == "object":
        for prop, details in openai_schema["properties"].items():
            if "allOf" in details:
                openai_schema["properties"][prop] = _pydantic_to_openai_schema(
                    details["allOf"][0]
                )
            elif details.get("type") == "array" and "items" in details:
                openai_schema["properties"][prop]["items"] = _pydantic_to_openai_schema(
                    details["items"]
                )
    
    # Add pattern if it exists
    if "pattern" in schema:
        openai_schema["pattern"] = schema["pattern"]

    return openai_schema

def pydantic_to_openai_definition(model: type[BaseModel]) -> dict:
    """Convert Pydantic schema to OpenAI schema."""

    schema = model.model_json_schema()

    openai_schema = {
        "properties": schema.get("properties", {}),
    }
    if "required" in schema:
        openai_schema["required"] = schema["required"]
    if "description" in schema:
        openai_schema["description"] = schema["description"]
    if schema["type"] == "object":
        for prop, details in openai_schema["properties"].items():
            if "allOf" in details:
                openai_schema["properties"][prop] = _pydantic_to_openai_schema(
                    details["allOf"][0]
                )
            elif details.get("type") == "array" and "items" in details:
                openai_schema["properties"][prop]["items"] = _pydantic_to_openai_schema(
                    details["items"]
                )
    
    # Add pattern if it exists
    if "pattern" in schema:
        openai_schema["pattern"] = schema["pattern"]

    return openai_schema