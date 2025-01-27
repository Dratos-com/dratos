################### Adding Project Root to Python Path #############################

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

################### Adding Project Root to Python Path #############################

from dratos import LLM, Agent
from dratos import OpenAIEngine

from pydantic import BaseModel


def generate(prompt):
    """
    Generate a text completion for a given prompt.

    >>> generate("Say hi in french")
    'Bonjour!'
    """

    class Capital(BaseModel):
        capital: str
        country: str

    llm = LLM(
        #model_name="llama3.2:latest", 
        model_name="deepseek-r1:14b",
        engine=OpenAIEngine(
            base_url="http://localhost:11434/v1",
            api_key="dummy-key",
        ),
    )

    simple_agent = Agent(
        name="simple_agent",
        verbose=True,
        response_model=Capital,
        response_validation=True,
        llm=llm,
    )

    return simple_agent.sync_gen({"text": prompt})

generate("What is the capital of Canada?")