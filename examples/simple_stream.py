"""
* WORKING

What this script does:
Generate a response from an LLM.

Requirements:
   - Add the folowing API key(s) in your .env file:
    - OPENAI_API_KEY

To run the script, run the following command in your terminal:
   streamlit run examples/stream_response_in_streamlit.py
"""

################### Adding Project Root to Python Path #############################

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

################### Adding Project Root to Python Path #############################

import asyncio

from dratos import LLM, prompt, Agent
from dratos import OpenAI

llm = LLM(
    model_name="gpt-4o", 
    engine=OpenAI(),
)

@prompt
def poem_prompt(country):
   """
   Write a short poem in markdown format about {{country}}
   """

agent_with_tool = Agent(
    name="agent_with_tool",
    llm=llm,
    verbose=True,
    markdown_response=True,
)

async def get_final_result():
    response = ""
    async for value in agent_with_tool.async_gen(poem_prompt("Canada")):
        response += value
    return response

final_result = asyncio.run(get_final_result())