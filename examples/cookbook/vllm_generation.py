"""
! Not working

What this script does:
Generate a response from an LLM using VLLM.

Requirements:
   - 

"""

################### Adding Project Root to Python Path #############################

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

################### Adding Project Root to Python Path #############################

import asyncio

from dratos import LLM, OpenAIEngine, OpenAIEngineConfig

oai_config = OpenAIEngineConfig(data={"api_key": "EMPTY", 
                                      "base_url": "http://localhost:8000/v1", 
                                      "temperature": 0.5, 
                                      "max_tokens": 4096, 
                                      "top_p": 1, 
                                      "frequency_penalty": 0, 
                                      "presence_penalty": 0}
                                      )
vllm_engine = OpenAIEngine(config=oai_config)

llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct", engine=vllm_engine)

response = asyncio.run(llm.generate("What is the capital of Canada?"))

print(response)