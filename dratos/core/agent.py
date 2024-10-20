from typing import List, Dict, Type, Any
import json
from dratos.models.types.LLM import LLM
from pydantic import BaseModel

from dratos.utils.utils import tool_definition, pydantic_to_openai_schema, extract_json_from_str

from dratos.models.providers.openai_engine import OpenAIEngine

from rich import print as rprint
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
from rich.style import Style
from rich.text import Text
from rich.live import Live
from rich.console import Console, Group


import tiktoken

import logging
from rich.logging import RichHandler




logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("rich")


# logger.info("This is an info message.")
# logger.warning("This is a warning message.")
# logger.error("This is an error message.")

class Agent:
    def __init__(
            self,
            name: str,
            llm: LLM,
            system_prompt: str = None,
            verbose: bool = False,
            history: bool = False,
            tools: List[Dict] = None,
            pass_results_to_llm: bool = False, # only with tools
            markdown_response: bool = False,
            response_model: BaseModel = None,
            response_validation: bool = False, # only with reponse_format
            completion_setting: Dict = {},
        ):
        self.name = name
        self.llm = llm
        self.completion_setting = completion_setting
        self.history = history
        self.tools = tools
        self.pass_results_to_llm = pass_results_to_llm
        self.response_model = response_model
        self.response_validation = response_validation
        self.markdown_response = markdown_response
        self.verbose = verbose

        logger.info(f"🎬 Initializing {name}...")

        self.messages = []

        if self.response_model and not self.llm.support_structured_output:
            self.messages.append({"role": "system", "content": 
                                    f"{system_prompt if system_prompt else ''}\
                                    \n{self.pydantic_schema_description()}"})
            
            self.pretty(self.messages[0]["content"], title="System prompt")

        if tools is not None and response_model is not None:
            raise ValueError("Cannot use both 'tools' and 'response_model'.")
        
        if pass_results_to_llm:
            if tools is None:
                raise ValueError("'pass_results_to_llm' requires 'tools'.")
        
        if response_validation:
            if response_model is None:
                raise ValueError("A response can only be validated if a `response_model` is provided.")

    def pretty(self, message: str, title: str):
        if not self.verbose:
            return
        if title == "Response":
            color = "green"
        elif title == "Prompt":
            color = "blue"
        elif title == "System prompt":
            color = "bright_black"
        else:
            color = "dark_orange3"

        try:
            tokenizer = tiktoken.encoding_for_model(self.llm.model_name)
            tokens = len(tokenizer.encode(str(message)))

        except KeyError:
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        
        title = f"{title} ({tokens} tokens)"

        if self.markdown_response:
            content = Markdown(str(message))
        else:
            try:
                json_data, start, end = extract_json_from_str(message)
                json_string = json.dumps(json_data, indent=4)
                json_content = Syntax(json_string, "json", theme="monokai", line_numbers=False)
                content = Group(
                Text(start),
                json_content,
                    Text(end)
                )
            except Exception as e:
                content = str(message)

        rprint(Panel(content, 
                    title=title, 
                    title_align="left", 
                    style=Style(color=color)
                    ))

    async def pretty_stream(self, prompt: str, messages: List[Dict], completion_setting: Dict):
        console = Console()
        try:
            tokenizer = tiktoken.encoding_for_model(self.llm.model_name)
        except KeyError:
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        
        async def stream():
            response = ""
            tokens = 0
            async for chunk in self.llm.async_gen(prompt,
                                    messages=self.messages, 
                                    **completion_setting):
                response += chunk
                tokens += len(tokenizer.encode(chunk))
                yield chunk, tokens, response
        
        if self.verbose:
            with Live(console=console, refresh_per_second=4) as live:
                async for chunk, tokens, response in stream():
                    live.update(
                            Panel(
                            Markdown(str(response)) if self.markdown_response else str(response),
                            title=f"Response ({tokens} tokens)",
                            title_align="left",
                            style=Style(color="green"),
                            expand=False
                        )
                    )
                    yield chunk
        else:
            async for chunk, _, _ in stream():
                yield chunk

    def append_message(self, message: str, role: str, **kwargs):
        return self.messages.append({"role": role, "content": message, "context": kwargs})
    
    def record_message(self, message: str, role: str, **kwargs):
        if role == "System prompt":
            content = message
        elif role == "Prompt":
            content = message
        elif role == "Response":
            content = message
        elif role == "Tool results":
            content = message
        elif role == "Tool call":
            for tool_call in message:
                content = {
                    "name": tool_call["name"],
                    "arguments": f'{tool_call["arguments"]}'
                } 
                kwargs.update({k: v for k, v in tool_call.items() if k not in ["name", "arguments"]})
                self.append_message(content, role, **kwargs)
                return
        else:
            raise ValueError(f"Unknown message role: {role}")
        
        self.append_message(content, role, **kwargs)
        self.pretty(content, role)

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

        async def generate():
            # Setup
            if not isinstance(prompt, str):
                raise ValueError("Prompt must be a string")

            self.record_message(prompt, role="Prompt")
            self.log_agent_info()

            completion_setting = kwargs if kwargs else self.completion_setting
            tools = [tool_definition(tool) for tool in self.tools] if self.tools else None

            # Generation
            response = await self.llm.sync_gen(
                #prompt=prompt,
                response_model=self.response_model,
                tools=tools,
                messages=self.messages[:-1] if self.history else [self.messages[-1]],
                **completion_setting)

            # Tool calling
            if self.tools and not isinstance(response, str):
                self.record_message(response, role="Tool call")
                for tool_call in response:
                    for tool in self.tools:
                        if tool.__name__ == tool_call["name"]:
                            result = tool(**tool_call["arguments"])
                            if self.pass_results_to_llm:
                                self.record_message(result, role="Tool results", **tool_call)
                                response = await self.llm.sync_gen(
                                    #prompt=result,
                                    response_model=None,
                                    tools=None,
                                    messages=self.messages[:-1] if self.history else self.messages[-2:],
                                    **completion_setting)
                                self.record_message(response, role="Response")
                            else:
                                return result
            else:
                self.record_message(response, role="Response")

            # Validation
            if self.response_validation:
                response = self.pydantic_validation(response)

            return response

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return loop.run_until_complete(generate())
        except RuntimeError:
            return asyncio.run(generate())

    async def async_gen(self, prompt: str, **kwargs):
        
        # Setup
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
        if self.tools or self.response_model:
            raise ValueError("Cannot use 'tools' and 'response_model' with async_gen, use sync_gen instead.")
        if kwargs:
            completion_setting = kwargs
        else:
            completion_setting = self.completion_setting

        self.record_message(prompt, role="Prompt")
        
        # Generation
        response = ""
        async for chunk in self.pretty_stream(prompt,
                                            messages=self.messages[:-1] if self.history else [], 
                                            completion_setting=completion_setting):
            response += chunk
            yield chunk

        self.record_message(response, role="Response")
    
    def pydantic_schema_description(self):
        response_model = pydantic_to_openai_schema(self.response_model)

        return f"Always respond following the specifications:\
                {json.dumps(pydantic_to_openai_schema(response_model))}\
                \nYour response will include all required properties in a Json format."

    def log_agent_info(self):
        tools_list = [tool.__name__ for tool in self.tools] if self.tools else None
        response_model_name = self.response_model.__name__ if self.response_model else None
        logger.info(f"Tools: {tools_list}")
        logger.info(f"Response Model: {response_model_name}")
