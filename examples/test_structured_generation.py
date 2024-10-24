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
from dratos import OpenAI

from pydantic import BaseModel

def structured_generation(prompt):
    """
    Generate a text completion for a given prompt.

    >>> structured_generation("What is the capital of Canada?")
    Capital(capital='Ottawa', country='Canada')
    """
    llm = LLM(
        model_name="gpt-4o-2024-08-06", 
        engine=OpenAI(),
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
        engine=OpenAI(),
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
