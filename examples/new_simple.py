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
from pydantic import BaseModel

# test_engine = OpenAIEngine()
# print(test_engine.get_completion_setting())

llm = LLM(
    model_name="gpt-4o-2024-08-06", 
    engine=OpenAIEngine(),
)

@prompt
def prompt_with_tool():
   """
   Please provide the answer in the following operation: 567890 + 5678909876 using the provided tool.
   """

def add(arg1, arg2):
    return arg1 + arg2

   
agent_with_tool = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant.",
    tools=[add],
    pass_results_to_llm=True,
    markdown_response=True,
)
#response = asyncio.run(agent_with_tool.generate(prompt_with_tool()))
# response = asyncio.run(agent_with_tool.generate("Say hi in french"))


class Capital(BaseModel):
    capital: str
    country: str

@prompt
def capital_prompt(country):
   """
   What is the capital of {{country}}?
   Respond in a markdown format.
   """

agent_without_tool = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant.",
    response_model=Capital,
    response_validation=True,
)
response = asyncio.run(agent_without_tool.generate(capital_prompt("Canada")))
#response = asyncio.run(agent_without_tool.generate("Say hello in Japanese"))

print(response)
