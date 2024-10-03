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

from dratos import LLM, OpenAIEngine, prompt


openai_engine = OpenAIEngine()

llm = LLM("gpt-4o", openai_engine)

@prompt
def capital_prompt(country):
   """
   What is the capital of {{country}}?
   """


response = asyncio.run(llm.generate(capital_prompt("Canada")))

print("\033[94m" + response + "\033[0m")