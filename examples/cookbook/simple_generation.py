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

import  dotenv
import asyncio

from dratos.models.obj.base_language_model import LLM
from dratos.models.serve.engines.openai_engine import OpenAIEngine, OpenAIEngineConfig

oai_config = OpenAIEngineConfig()

openai_engine = OpenAIEngine(config=oai_config)

llm = LLM("gpt-4o", openai_engine)

response = asyncio.run(llm.generate("What is the capital of Canada?"))

print("\033[94m" + response + "\033[0m")