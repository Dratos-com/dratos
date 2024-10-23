from typing import List, Dict, Type
import json
from dratos.models.types.LLM import LLM
from pydantic import BaseModel

from dratos.utils.utils import function_to_openai_definition, pydantic_to_openai_definition, extract_json_from_str
from dratos.utils.pretty import pretty, pretty_stream
from dratos.memory.memory import Memory

import logging
from rich.logging import RichHandler


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("rich")


class Agent:
    def __init__(
            self,
            name: str,
            llm: LLM,
            memory_config: Dict = None,
            memory: Memory = None,
            system_prompt: str = None,
            verbose: bool = False,
            history: bool = False,
            tools: List[Dict] = None,
            markdown_response: bool = False,
            response_model: BaseModel = None,
            response_validation: bool = False, # only with reponse_model
            completion_setting: Dict = {},
        ):
        self.name = name
        self.llm = llm
        self.memory_config = memory_config
        self.completion_setting = completion_setting
        self.history = history
        self.tools = tools
        self.response_model = response_model
        self.response_validation = response_validation
        self.markdown_response = markdown_response
        self.verbose = verbose
        self.system_prompt = system_prompt
          
        self.messages = []

        # Logging
        logger.info(f"🎬 Initializing {name}...")

        if tools is not None and response_model is not None:
            raise ValueError("Cannot use both 'tools' and 'response_model'.")
        
        if response_validation:
            if response_model is None:
                raise ValueError("A response can only be validated if a `response_model` is provided.")

        if memory:
            self.memory = memory
        elif memory_config:
            logger.info(f"🔍 Initializing Memory from config")
            self.memory = Memory.from_config(memory_config)
        else:
            self.memory = None  

        # System prompt
        if self.response_model and not self.llm.support_structured_output:
            system_prompt = f"{system_prompt}\n"
            self.record_message(f"{system_prompt or ''}\n{self.response_model_defition()}", "System prompt")
        elif self.tools and not self.llm.support_tools:
            self.record_message(f"{system_prompt or ''}\n{self.tool_definition()}", "System prompt")
        elif system_prompt:
            self.record_message(system_prompt, "System prompt")

    def append_message(self, message: str, role: str, **kwargs):
        return self.messages.append({"role": role, "content": message, "context": kwargs})
    
    def record_message(self, message: str, role: str, verbose: bool = True, **kwargs):
        if role == "System prompt":
            content = message
        elif role == "Prompt":
            content = message
        elif role == "Response":
            content = message
        elif role == "Tool call":
            content = {
                "name": message["name"],
                "arguments": f'{message["arguments"]}'
            } 
            kwargs.update({k: v for k, v in message.items() if k not in ["name", "arguments"]})
        else:
            raise ValueError(f"Unknown message role: {role}")
        
        self.append_message(content, role, **kwargs)
        pretty(self, content, role) if verbose else None

    def pydantic_validation(self, response:str)-> Type[BaseModel]:
        """Validates the response using Pydantic."""
        parsed_json, _, _ = extract_json_from_str(response)
        parameters = json.dumps(parsed_json)
        try:
            model = self.response_model.__pydantic_validator__.validate_json(parameters, strict=True)
            logger.info(f"✅ Response format is valid")
            return model
        except Exception as e:
            logger.error(f"❌ Response format is not valid: {e}")
            raise e
    
    def sync_gen(self, prompt: str, **kwargs):
        import asyncio

        async def generate(prompt: str):
            # Setup
            if not isinstance(prompt, str):
                raise ValueError("Prompt must be a string")

            prompt = self.search_memory(prompt)

            self.record_message(prompt, role="Prompt")
            self.log_agent_info()

            completion_setting = kwargs if kwargs else self.completion_setting
            tools = self.tool_definition()

            # Generation
            response = await self.llm.sync_gen(
                response_model=self.response_model,
                tools=tools,
                messages=self.get_messages(),
                **completion_setting)

            # Tool calling
            if self.tools and not isinstance(response, str):
                complete_result = dict()
                for tool_call in response:
                    for tool in self.tools:
                        if tool.__name__ == tool_call["name"]:
                            result = tool(**tool_call["arguments"])
                            complete_result.update({tool_call["name"]: result})
                self.record_message(complete_result, role="Response")
                return complete_result
            else:
                self.record_message(response, role="Response")

            # Validation
            if self.response_validation:
                response = self.pydantic_validation(response)

            return response

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return loop.run_until_complete(generate(prompt))
        except RuntimeError:
            return asyncio.run(generate(prompt))

    async def async_gen(self, prompt: str, **kwargs):
        
        # Setup
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
        if self.tools or self.response_model:
            raise ValueError("Cannot use 'tools' and 'response_model' with async_gen, use sync_gen instead.")
        
        completion_setting = kwargs if kwargs else self.completion_setting

        if self.memory:
            result = self.memory.search(query=prompt, agent_id=self.name)
            prompt = f"{prompt}\n\n Related memories:\n{result}"

        self.record_message(prompt, role="Prompt")
        
        # Generation
        response = ""
        async for chunk in pretty_stream(self,
                                         messages=self.get_messages(), 
                                         completion_setting=completion_setting):
            response += chunk
            yield chunk

        self.record_message(response, role="Response", verbose=False)
    
    def response_model_defition(self):
        response_model = pydantic_to_openai_definition(self.response_model)
        return f"""Always respond following the specifications:
                    {json.dumps(response_model)}
                    \nYour response will include all required properties in a Json format."""
            
    def tool_definition(self):
        return [function_to_openai_definition(tool) for tool in self.tools] if self.tools else None

    def log_agent_info(self):
        tools_list = [tool.__name__ for tool in self.tools] if self.tools else None
        response_model_name = self.response_model.__name__ if self.response_model else None
        logger.info(f"Tools: {tools_list}")
        logger.info(f"Response Model: {response_model_name}")

    def get_messages(self):
        if self.history:
            return self.messages
        
        has_system_prompt = (
            (self.response_model and not self.llm.support_structured_output) or
            (self.tools and not self.llm.support_tools) or
            self.system_prompt
        )

        return [self.messages[0], self.messages[-1]] if has_system_prompt else [self.messages[-1]]

    def get_memory(self):
        return self.memory.get_all(agent_id=self.name) if self.memory else None
    
    def add_memory(self, memory: str, **kwargs):
        if self.memory:
            return self.memory.add(memory, agent_id=self.name, metadata=kwargs)
        else:
            logger.info("Initializing Memory Instance")
            self.memory = Memory()
            return self.memory.add(memory, agent_id=self.name, metadata=kwargs)

    def search_memory(self, query: str):
        if self.memory:
            # result = self.memory.search(query=query, agent_id=self.name)
            # related_memories = '\n'.join([f"{memory['created_at']} - {memory['memory']}" for memory in result])
            results = self.memory.search(query, agent_id=self.name)
            related_memories = '\n'.join([f"{result['text']}" for result in results])
            return f"{query}\n\nMemory:\n{related_memories}"
        else:
            return None