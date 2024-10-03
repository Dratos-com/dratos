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

import asyncio

from dratos import LLM, prompt, Agent
from dratos import OpenAIEngine

llm = LLM(
    model_name="gpt-4o", 
    engine=OpenAIEngine(),
)

@prompt
def poem_prompt(country):
   """
   Write a short poem about {{country}}
   """


agent_with_tool = Agent(
    name="agent_with_tool",
    llm=llm,
    # markdown_response=True,
    verbose=True,
)

async def get_final_result():
    result = []
    async for value in agent_with_tool.async_gen(poem_prompt("Canada")):
        print(value)
        result.append(value)
    return ''.join(result)

final_result = asyncio.run(get_final_result())

# agent_with_tool.sync_gen(poem_prompt("Canada"))