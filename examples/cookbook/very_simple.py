"""
* WORKING

What this script does:
Generate a structured response from an LLM.

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

from dratos.models.obj.base_language_model import LLM

llm = LLM()

response = asyncio.run(llm.generate_structured("What is the capital of Canada?", structure="{location: str, capital: str}"))

print("\033[94m" + response + "\033[0m")