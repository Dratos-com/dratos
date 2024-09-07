class OrchestrationManager(Manager):
    def __init__(self):
        ray.init()

    def create_agent(self, model_name: str) -> Agent:
        model = PreTrainedModel.from_pretrained(model_name)
        tokenizer = PreTrainedTokenizer.from_pretrained(model_name)
        return Agent.remote(model, tokenizer)

    def create_tool(self, func: Callable) -> Tool:
        return Tool.remote(func)

    def process_data(
        self, agent: Agent, data: Union[StructuredData, UnstructuredData]
    ) -> Any:
        return ray.get(agent.process.remote(data))

    def run_tool(self, tool: Tool, *args, **kwargs) -> Any:
        return ray.get(tool.run.remote(*args, **kwargs))
