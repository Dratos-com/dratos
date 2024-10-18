import json
import torch
import inspect
from pydantic import BaseModel
from typing import Callable, Dict, Any

def tool_definition(tool: Callable) -> Dict:
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

def tool_result( args: Dict, result: Any, id: str) -> Dict:
    content =  {
        **{k:v for k,v in args.items()}, 
        **{"result": result}
    }
    return {
        "role": "tool",
        "content": content.__str__(),
        "tool_call_id": id
    }

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

def get_device():
    """
    This module provides a function to determine the appropriate device for PyTorch based on the available backend.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.backends.cuda.is_built():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def pydantic_to_openai_schema(model: type[BaseModel]) -> dict:
    """Convert Pydantic model to OpenAI schema."""
    schema = model.model_json_schema()
    return _pydantic_to_openai_schema(schema)

def _pydantic_to_openai_schema(schema: dict) -> dict:
    """Convert Pydantic schema to OpenAI schema with required flag in each property."""
    openai_schema = {}
    
    required_fields = set(schema.get("required", []))  # Convert required list to a set for easy lookup
    
    if schema["type"] == "object":
        for prop, details in schema.get("properties", {}).items():
            # Recursively handle nested properties if necessary
            if "allOf" in details:
                prop_schema = _pydantic_to_openai_schema(details["allOf"][0])
            elif details.get("type") == "array" and "items" in details:
                prop_schema = _pydantic_to_openai_schema(details["items"])
            else:
                prop_schema = details  # Copy the details as is

            # Add whether the property is required directly within its definition
            prop_schema["required"] = prop in required_fields
            
            # Add the property to the main schema
            openai_schema[prop] = prop_schema
    
    # Add pattern if it exists at the root level
    if "pattern" in schema:
        openai_schema["pattern"] = schema["pattern"]

    return openai_schema