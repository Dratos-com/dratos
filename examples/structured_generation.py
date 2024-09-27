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

import  dotenv
import asyncio

from dratos.models.obj.base_language_model import LLM
from dratos.models.serve.engines.openai_engine import OpenAIEngine, OpenAIEngineConfig

dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

oai_config = OpenAIEngineConfig(data={"api_key": api_key, 
                                      "base_url": "https://api.openai.com/v1", 
                                      "temperature": 0.5, 
                                      "max_tokens": 3000, 
                                      "top_p": 1, 
                                      "frequency_penalty": 0, 
                                      "presence_penalty": 0})

openai_engine = OpenAIEngine(config=oai_config)

llm = LLM("gpt-4o", openai_engine)

response = asyncio.run(llm.generate_structured("What is the capital of Canada?", structure="{location: str, capital: str}"))

print(response)