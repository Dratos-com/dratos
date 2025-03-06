import json
import inspect
from pydantic import BaseModel
from typing import Callable, Dict

from pydantic_core import from_json


def extract_json_from_str(response: str) -> tuple[dict, str, str, bool]:
    """
    Extracts a JSON object from a string.

    Args:
        response (str): The string containing the JSON object.

    Returns:
        leading string, json object, trailing string, partial Json (Boolean)

    Raises:
        ValueError: If the string does not contain a valid JSON object.
    """
    try:
        json_start = response.index("{")
        json_end = response.rfind("}")
        return json.loads(response[json_start : json_end + 1]), response[0:json_start], response[json_end + 1:], False # full Json
    except Exception:
        try:
            return from_json(response[json_start :], allow_partial=True), response[0:json_start], "", True # partial Json
        except Exception as e:
            raise ValueError("No valid JSON structure found in the input string: " + str(e))

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

def pydantic_to_openai_definition(model: type[BaseModel]) -> dict:
    """Convert Pydantic schema to OpenAI schema."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    
    def resolve_schema(sub_schema: dict) -> dict:
        """Resolve schema references and convert to OpenAI format."""
        if "$ref" in sub_schema:
            # Extract the definition name from reference (e.g., "#/$defs/Row" -> "Row")
            ref_name = sub_schema["$ref"].split("/")[-1]
            if ref_name in defs:
                return resolve_schema(defs[ref_name])
            return {}

        result = {
            "properties": sub_schema.get("properties", {}),
        }
        
        if "required" in sub_schema:
            result["required"] = sub_schema["required"]
        if "description" in sub_schema:
            result["description"] = sub_schema["description"]
        if "type" in sub_schema:
            result["type"] = sub_schema["type"]
        if "pattern" in sub_schema:
            result["pattern"] = sub_schema["pattern"]

        if sub_schema.get("type") == "object":
            for prop, details in result["properties"].items():
                if details.get("type") == "array" and "items" in details:
                    result["properties"][prop]["items"] = resolve_schema(details["items"])

        return result

    return resolve_schema(schema)