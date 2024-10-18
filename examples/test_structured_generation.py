"""
* WORKING

What this script does:
Generate a tool request using an LLM.

Requirements:
Add the folowing API key(s) in your .env file:
   - OPENAI_API_KEY
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
    )

    return agent.sync_gen(prompt)

# print(structured_generation("What is the capital of Canada?"))
