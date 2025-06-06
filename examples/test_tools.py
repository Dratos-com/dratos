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


def tool(prompt):
    """
    Generate a text completion for a given prompt.

    >>> tool("What is 2 + 2?")
    {'add': 4}
    """
    llm = LLM(
        model_name="gpt-4o", 
        engine=OpenAIEngine(),
    )

    def add(arg1: int, arg2: int) -> int:
        return arg1 + arg2

    agent = Agent(
        name="agent",
        llm=llm,
        #verbose=True,
        tools=[add]
    )

    return agent.sync_gen({"text": prompt})

#tool("What is 2 + 2?")


def use_multiple_tools(prompt):
    """
    Generate a text completion for a given prompt using multiple tools.

    >>> use_multiple_tools("What is 2 + 2 and 3 * 3?")
    {'add': 4, 'multiply': 9}
    """
    llm = LLM(
        model_name="gpt-4o", 
        engine=OpenAIEngine(),
    )

    def add(arg1: int, arg2: int) -> int:
        return arg1 + arg2

    def multiply(arg1: int, arg2: int) -> int:
        return arg1 * arg2

    agent = Agent(
        name="agent",
        llm=llm,
        #verbose=True,
        tools=[add, multiply],
    )

    return agent.sync_gen({"text": prompt})

#use_multiple_tools("What is 2 + 2 and 3 * 3?")