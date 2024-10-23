"""
* WORKING

What this script does:
Generate a tool request using an LLM.

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

from dratos import LLM, Agent
from dratos import OpenAI, QdrantMemory

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# def memory(prompt):
#     """
#     Generate a text completion for a given prompt.

#     >>> memory("What do I like to do on weekends?")
    
#     """
#     llm = LLM(
#         model_name="gpt-4o", 
#         engine=OpenAI(),
#     )

#     memory_config = {
#         "vector_store": {
#             "provider": "qdrant",
#             "config": {
#                 "collection_name": "test_memory",
#                 "host": "localhost",
#                 "port": 6333,
#             }
#         }
#     }

#     agent = Agent(
#         name="agent",
#         llm=llm,
#         verbose=True,
#         memory_config=memory_config
#     )
#     agent.get_memory()
#     agent.add_memory("I like to take long walks on weekends.", metadata={"category": "hobbies"})
    
#     return agent.sync_gen(prompt)

def qdrant_memory(prompt:str):
    collection_name = "my_test_collection"
    memory = QdrantMemory(collection_name)

    # Add documents
    documents = [
        "I like macha tea",
        "I hate coffee",
        "I like to play the guitar"
    ]

    memory.add(documents, agent_id="agent")

    # Search
    query = "macha tea"
    results = memory.search(query, agent_id="agent")

    for result in results:
        print(f"ID: {result['id']}, Score: {result['score']}, Text: {result['text']}")

    llm = LLM(
        model_name="gpt-4o", 
        engine=OpenAI(),
    )

    agent = Agent(
        name="agent",
        memory=memory,
        llm=llm,
        verbose=True
    )
    return agent.sync_gen(prompt)

with mlflow.start_run():
   mlflow.openai.autolog()
   # print(memory("What do I like to do?"))
   print(qdrant_memory({"text": "What do I like to do?"}))
