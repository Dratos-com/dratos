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

import streamlit as st

# Title of the app
st.title("Streaming Text with Streamlit")

# Create a placeholder for streaming text
placeholder = st.empty()


llm = LLM(
    model_name="gpt-4o", 
    engine=OpenAI(),
)

@prompt
def poem_prompt(country):
   """
   Write a very short poem about {{country}}
   """

agent_with_tool = Agent(
    name="agent_with_tool",
    llm=llm,
    verbose=True,
)

async def get_final_response():
    response = ""
    async for value in agent_with_tool.async_gen(poem_prompt("Canada")):
        response += value
        placeholder.text(response)
    return response

final_response = asyncio.run(get_final_response())
