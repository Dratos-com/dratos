"""
TODO IN PROGRESS

What this script does:
Generate a response from an LLM locally.

Requirements:
    - install llama.cpp `pip install llama-cpp-python`
"""

################### Adding Project Root to Python Path #############################

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

################### Adding Project Root to Python Path #############################

import asyncio

from dratos import LLM, OpenAIEngineConfig, LlamaCppEngine

oai_config = OpenAIEngineConfig(data={"api_key": "EMPTY", 
                                      "base_url": "https://api.openai.com/v1", 
                                      "temperature": 0.5, 
                                      "max_tokens": 3000, 
                                      "top_p": 1, 
                                      "frequency_penalty": 0, 
                                      "presence_penalty": 0})

llama_engine = LlamaCppEngine(config=oai_config)

llm = LLM("llama-7b", llama_engine)

response = asyncio.run(llm.generate("What is the capital of Canada?"))

print(response)