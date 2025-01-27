################### Adding Project Root to Python Path #############################

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

################### Adding Project Root to Python Path #############################

from dratos import LLM, Agent
from dratos import OpenAIEngine

from pydantic import BaseModel, Field
from typing import List

def generate(prompt):
    """
    Generate a text completion for a given prompt.

    >>> generate("Say hi in french")
    'Bonjour!'
    """

    class Row(BaseModel):
        row_id: int = Field(..., description="Index of the row.")
        cluster_id: int = Field(..., description="Index of the operation. By default, each row is its own cluster (Same index as the row index).")

    class Clusters(BaseModel):
        clusters: List[Row] = Field(..., description="Assigns a cluster index to each row. List ALL rows.")


    llm = LLM(
        #model_name="llama3.2:latest", 
        model_name="deepseek-r1:14b",
        engine=OpenAIEngine(
            base_url="http://localhost:11434/v1",
            api_key="dummy-key",
        ),
    )

    group_rows = Agent(
        name="group_rows",
        llm=llm,
        response_model=Clusters,
        response_validation=True,
        verbose=True,
    )

    return group_rows.sync_gen({"text": prompt})

# generate("""
# Row 1: Operation A
# Row 2: Operation B
# Row 3: Operation C
# Row 4: Operation A
# Row 5: Operation E
# Row 6: Operation D
# Row 7: Operation G
# Row 8: Operation D
# Row 9: Operation D
# Row 10: Operation B
# """)


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

#generate("What is the capital of Canada?")


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
        model_name="deepseek-r1-distill-llama-70b",
        engine=OpenAIEngine(
            base_url="https://api.groq.com/openai/v1/",
            api_key="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",
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