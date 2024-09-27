"""
TODO how to use mlflow's ui?

What this script does:
Generate a response from an LLM and log the response to mlflow.

Requirements:
   - Add the folowing API key(s) in your .env file:
      - OPENAI_API_KEY

"""

################### Adding Project Root to Python Path #############################

import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd()))
sys.path.append(project_root)

################### Adding Project Root to Python Path #############################

import  dotenv
import mlflow
import asyncio

from dratos.models.obj.base_language_model import LLM
from dratos.models.serve.engines.openai_engine import OpenAIEngine, OpenAIEngineConfig

dotenv.load_dotenv()

mlflow_client = mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
api_key = os.getenv("OPENAI_API_KEY")

oai_config = OpenAIEngineConfig(data={"api_key": api_key, 
                                      "base_url": "https://api.openai.com/v1", 
                                      "temperature": 0.5, 
                                      "max_tokens": 3000, 
                                      "top_p": 1, 
                                      "frequency_penalty": 0, 
                                      "presence_penalty": 0})

openai_engine = OpenAIEngine(config=oai_config)

with mlflow.start_run():
   mlflow.openai.autolog()
   mlflow.set_experiment("OpenAi")
   mlflow.set_tag("model", "gpt-4o")
   mlflow.set_tag("prompt", "What is the capital of Canada?")

   llm = LLM("gpt-4o", openai_engine)

   response = asyncio.run(llm.generate("What is the capital of Canada?"))

print("\033[94m" + response + "\033[0m")

# # Setup traces
# import mlflow
# @mlflow.trace