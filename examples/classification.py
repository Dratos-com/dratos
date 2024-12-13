################### Adding Project Root to Python Path #############################

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

################### Adding Project Root to Python Path #############################

from dratos import LLM, Agent
from dratos import OpenAIEngine

from pydantic import BaseModel, Field, field_validator
from enum import Enum



def generate(prompt):
    """
    Generate a text completion for a given prompt.

    >>> generate("Say hi in french")
    'Bonjour!'
    """

    categories = [
        {"category": "A", "description": "Don't choose me"},
        {"category": "B", "description": "Don't choose me"},
        {"category": "C", "description": "Don't choose me"},
        {"category": "D", "description": "Choose me"},
        {"category": "E", "description": "Don't choose me"},
        {"category": "F", "description": "Don't choose me"},
    ]

    # Define the Pydantic model
    class Classification(BaseModel):
        category: str = Field(
            ..., 
            options=[cat["category"] for cat in categories],
            description="\n".join(f"{cat['category']}: {cat['description']}" for cat in categories)
        )

    llm = LLM(
        model_name="gpt-4o-2024-08-06", 
        engine=OpenAIEngine(),
    )

    simple_agent = Agent(
        name="simple_agent",
        verbose=True,
        response_model=Classification,
        response_validation=True,
        llm=llm,
    )

    return simple_agent.sync_gen({"text": prompt})

generate("What is the option to choose?")