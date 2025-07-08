"""
* WORKING

What this script does:
Generate a tool request using an LLM.

Requirements:
Add the folowing API key(s) in your .env file:
   - OPENAI_API_KEY


BE AWARE:
Using OpenAI built-in response_model support, only the following types are supported:
String
Number
Boolean
Object
Array
Enum
anyOf
This is in line with JSON data type support (which doesn't have support for datetime).
"""

import sys
import os
import asyncio

# Add project root to path
project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

from dratos import LLM, Agent, LiteLLMEngine
from pydantic import BaseModel

class Capital(BaseModel):
    capital: str
    country: str

class Europe(BaseModel):
    countries: list[Capital]

async def main():
    llm = LLM(model_name="vertex_ai/gemini-1.5-flash", 
              engine=LiteLLMEngine(provider="vertex_ai"))

    agent = Agent(
        name="agent",
        llm=llm,
        response_model=Europe,
        response_validation=True,
        verbose=True,
    )

    response = ""
    async for value in agent.async_gen({"text": "Give me all country and capital pairs in Europe"}):
        response += value
    
    return response

if __name__ == "__main__":
    asyncio.run(main())
