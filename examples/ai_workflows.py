engine = OpenAIEngine(stream=False) # Default to True

# Setting up the LLM
llm = LLM(
    model_name="gpt-4o", 
    engine=engine,
)

# Prompt
@prompt()
def my_prompt(arg1, arg2):
    """
    You are a helpful assistant {{arg1}} + {{arg2}}.
    """

completion_setting = {
    "max_tokens": 1024,
    "temperature": 0.5,
    "top_p": 0.9,
    "top_k": 1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "stop": ["\n"],
}

# Define tools

def my_tool(arg1, arg2):
    return arg1 + arg2

# Define an output structure
class UserInfo(BaseModel):
    id: str
    uri: str
    name: str
    description: str
    tags: List[str]
    metadata: Dict[str, Any]


# Setting up an agent
my_first_agent = Agent(
    llm=llm,
       completion_setting=completion_setting,
    tools=[],
)

my_first_agent.generate(
    prompt=my_prompt,
    images="file/path/to/image" or binary data,
    )



class MemoryModel(BaseModel):
    # structured data
    "name": "my_memory",
    "description": "This is a memory",
    "last_contact": 12/31/2023,
    "ceo": True,
    # metadata
    "last_updated": 12/31/2023,
    "created": 12/31/2023,
    "created_by": "user_1",
    "updated_by": "user_1",
    "tags": ["tag_1", "tag_2"]
}


memory_1 = Memory(uri="/path/to/memory",)
memory_2 = Memory(
    uri="/path/to/memory", 
    embedding_model = "text-embedding-ada-002", 
    schema=MemoryModel)


# Load artifacts
my_artifact = "s3://bucket/path/to/artifact" or binary data or text data


memory_1.add(my_artifact)
memory_1.delete(my_artifact)
memory_1.get(my_artifact)
memory_1.name = "client_1"

agent_1 = Agent(
    llm=llm,
    tools=[],
    memory=[memory_1],
)


