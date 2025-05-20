"""
* WORKING

What this script does:
Generate a response from an LLM.

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
from dratos import GoogleEngine
from pydantic import BaseModel, Field
from typing import List, Dict

def generate(prompt):
    """
    Generate a text completion for a given prompt.

    >>> generate("Say hi in french")
    'Bonjour!'
    """
    llm = LLM(
        model_name="gemini-2.0-flash", 
        engine=GoogleEngine(),
    )
    from pydantic import BaseModel, Field
    from typing import List

    class Tables(BaseModel):
        tables: List[str] = Field(
            ...,
            description="""
            The provided data
            """
        )

    table_extraction_agent = Agent(
        name="table_extraction_agent",
        llm=llm,
        verbose=True,
        response_model=Tables,
        response_validation=True,
        continue_if_partial_json_response=True
    )

    return table_extraction_agent.sync_gen({"text": prompt})

generate("""[
  {
    "date": "2025-04-01",
    "revenue": 120000,
    "expenses": 75000,
    "net_income": 45000
  },
  {
    "date": "2025-04-02",
    "revenue": 125000,
    "expenses": 80000,
    "net_income": 45000
  },
  {
    "date": "2025-04-03",
    "revenue": 130000,
    "expenses": 85000,
    "net_income": 45000
  },
  {
    "date": "2025-04-04",
    "revenue": 110000,
    "expenses": 70000,
    "net_income": 40000
  },
  {
    "date": "2025-04-05",
    "revenue": 135000,
    "expenses": 82000,
    "net_income": 53000
  }
]
""")