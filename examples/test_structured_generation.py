"""
* WORKING

What this script does:
Generate a tool request using an LLM.

Requirements:
Add the folowing API key(s) in your .env file:
   - OPENAI_API_KEY


BE AWARE:
Using OpenAI built-in response_model support, only the following types are supported:
String
Number
Boolean
Object
Array
Enum
anyOf
This is in line with JSON data type support (which doesn’t have support for datetime).
"""

################### Adding Project Root to Python Path #############################

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

################### Adding Project Root to Python Path #############################

from dratos import LLM, Agent
from dratos import OpenAIEngine

from pydantic import BaseModel

def structured_generation(prompt):
    """
    Generate a text completion for a given prompt.

    >>> structured_generation("What is the capital of Canada?")
    Capital(capital='Ottawa', country='Canada')
    """
    llm = LLM(
        model_name="gpt-4o-2024-08-06", 
        engine=OpenAIEngine(),
    )

    class Capital(BaseModel):
        capital: str
        country: str

    agent = Agent(
        name="agent",
        llm=llm,
        response_model=Capital,
        response_validation=True,
        # verbose=True,
    )

    return agent.sync_gen({"text": prompt})

# structured_generation("What is the capital of Canada?")


def structured_generation_without_support(prompt):
    """
    Generate a text completion for a given prompt for an LLM that does not support structured generation.

    >>> structured_generation("What is the capital of Canada?")
    Capital(capital='Ottawa', country='Canada')
    """

    llm_without_structured_generation_support = LLM(
        model_name="gpt-4o", 
        engine=OpenAIEngine(),
    )
    
    class Capital(BaseModel):
        capital: str
        country: str

    agent = Agent(
        name="agent",
        llm=llm_without_structured_generation_support,
        response_model=Capital,
        response_validation=True,
        #verbose=True,
    )

    return agent.sync_gen({"text": prompt})

# structured_generation_without_support("What is the capital of Canada?")


def structured_generation_with_schema(prompt):
    """
    Generate a text completion for a given prompt.

    >>> structured_generation_with_schema("respond with synthetic data")
    Schema(id=8472, name='Jackson Lee', isActive=True, score=92.5, address=NestedModel0(street='123 Elm Street', city='San Francisco', zip='94102', coordinates=NestedModel1(latitude=37.7749, longitude=-122.4194)), preferences=NestedModel2(theme='dark', notifications=True), tags=['premium', 'new user', 'verified'], history=[NestedModel3(action='login', timestamp='2023-10-19T15:23:01'), NestedModel3(action='update_settings', timestamp='2023-10-18T14:05:13'), NestedModel3(action='purchase', timestamp='2023-10-17T18:44:09')], settings=NestedModel4(privacy=NestedModel5(shareData=False, trackLocation=True), security=NestedModel6(twoFactorAuth=True, backupCodes=['xyz-123', 'abc-789', 'def-456'])))
    """
    llm = LLM(
        model_name="gpt-4o-2024-08-06", 
        engine=OpenAIEngine(),
    )

    TestSchema = {
        "type": "object",
        "properties": {
            "id": { "type": "integer" },
            "name": { "type": "string" },
            "isActive": { "type": "boolean" },
            "score": { "type": "number" },
            "address": {
            "type": "object",
            "properties": {
                "street": { "type": "string" },
                "city": { "type": "string" },
                "zip": { "type": "string", "pattern": "^[0-9]{5}$" },
                "coordinates": {
                "type": "object",
                "properties": {
                    "latitude": { "type": "number", "minimum": -90, "maximum": 90 },
                    "longitude": { "type": "number", "minimum": -180, "maximum": 180 }
                },
                "required": ["latitude", "longitude"]
                }
            },
            "required": ["street", "city", "zip", "coordinates"]
            },
            "preferences": {
            "type": "object",
            "properties": {
                "theme": { "type": "string", "enum": ["light", "dark"] },
                "notifications": { "type": "boolean" }
            },
            "required": ["theme", "notifications"]
            },
            "tags": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 1,
            },
            "history": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                "action": { "type": "string" },
                "timestamp": { "type": "string", "format": "date-time" }
                },
                "required": ["action", "timestamp"]
            }
            },
            "settings": {
            "type": "object",
            "properties": {
                "privacy": {
                "type": "object",
                "properties": {
                    "shareData": { "type": "boolean" },
                    "trackLocation": { "type": "boolean" }
                },
                "required": ["shareData", "trackLocation"]
                },
                "security": {
                "type": "object",
                "properties": {
                    "twoFactorAuth": { "type": "boolean" },
                    "backupCodes": {
                    "type": "array",
                    "items": { "type": "string" },
                    "minItems": 5
                    }
                },
                "required": ["twoFactorAuth", "backupCodes"]
                }
            },
            "required": ["privacy", "security"]
            }
        },
        "required": ["id", "name", "isActive", "score", "address", "preferences", "tags", "history", "settings"]
        }

    agent = Agent(
        name="agent",
        llm=llm,
        response_schema=TestSchema,
        response_validation=True,
        # verbose=True,
    )

    return agent.sync_gen({"text": prompt})

# structured_generation_with_schema("respond with synthetic data")
