
class Agent:
    pass 

    # Composed of data, models, tools
    # input: prompt
    # outputs: response

    def __init__(self, 
                 name: str,                 
                 tools: List[Tool], 
                 llm: BaseLanguageModel, # LLMs
                 memory: BaseMemory,
                 artifacts: List[Artifact],
                 embedding_model: BaseEmbeddingModel,
                 ):
        
        self.name = name
        self.tools = tools
        self.llm = llm
        self.memory = memory
        self.artifacts = artifacts
        self.embedding_model = embedding_model
        




    def run(self, prompt: str):
        pass

