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
from dratos import OpenAIEngine


def generate(prompt):
    """
    Generate a text completion for a given prompt.

    >>> generate("Say hi in french")
    'Bonjour!'
    """
    llm = LLM(
        model_name="gpt-4o", 
        engine=OpenAIEngine(),
    )

    simple_agent = Agent(
        name="simple_agent",
        # verbose=True,
        llm=llm,
    )

    return simple_agent.sync_gen({"text": prompt})

# generate("Say hi in french")