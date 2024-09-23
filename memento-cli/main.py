import os
import asyncio
from beta.agents.agent import Agent
from beta.models.serve.engines.openai_engine import OpenAIEngine, OpenAIEngineConfig
from dotenv import load_dotenv

print("=============================")
isloaded = load_dotenv("../beta/.env")
print("isloaded=", isloaded)

print("OPENAI_API_KEY=", os.environ["OPENAI_API_KEY"])

# Set the OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# Initialize the OpenAI engine configuration
oai_config = OpenAIEngineConfig(data={
    "api_key": os.environ["OPENAI_API_KEY"],
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 150,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
})

# Initialize the OpenAI engine
openai_engine = OpenAIEngine(config=oai_config)

# Initialize the Agent with the OpenAI engine
agent = Agent(name="ConversationAgent", memory_db_uri="./memory/lancedb", git_repo_path="./memory/memories_repo")
agent.engine = openai_engine


async def main():
    conversation_id = "example_conversation"
    user_message = "What is the capital of Canada?"

    # Process the message with the agent
    response = await agent.process_message(conversation_id, user_message, "user")

    # Print the response
    print("Agent Response:", response["content"])

# Run the main function
asyncio.run(main())
