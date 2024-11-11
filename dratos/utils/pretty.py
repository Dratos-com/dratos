from typing import Dict, List, TYPE_CHECKING, Any

# Only import Agent for type checking
if TYPE_CHECKING:
    from dratos.core.agent import Agent
from dratos.utils.utils import extract_json_from_str

import logging

all_packages_installed = True
try:
    import tiktoken
    import json

    from rich import print as rprint
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.style import Style
    from rich.text import Text
    from rich.live import Live
    from rich.console import Console, Group
except ImportError:
    all_packages_installed = False

def get_tokens(agent: "Agent", text:str) -> int:
    if not all_packages_installed:
        return 0
    try:
        tokenizer = tiktoken.encoding_for_model(agent.llm.model_name)
        tokens = len(tokenizer.encode(str(text)))

    except KeyError:
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        tokens = len(tokenizer.encode(str(text)))
    return tokens

def pretty(agent: "Agent", message: Dict[str, Any], title: str):
    if not agent.verbose or not all_packages_installed:
        if agent.verbose:
            logging.info(f"tiktoken and rich libraries are required to see verbose output.")
        return
    if title == "Response":
        color = "green"
    elif title == "Prompt":
        color = "blue"
    elif title == "System prompt":
        color = "bright_black"
    else:
        color = "dark_orange3"

    tokens = get_tokens(agent, message["text"])
    title = f"{agent.name}: {title} ({tokens} tokens)" if tokens else f"{agent.name}: {title}"

    if agent.markdown_response:
        content = Markdown(str(message["text"]))
    else:
        try:
            json_data, start, end = extract_json_from_str(message["text"])
            json_string = json.dumps(json_data, indent=4)
            json_content = Syntax(json_string, "json", theme="monokai", line_numbers=False)
            content = Group(
            Text(start),
            json_content,
                Text(end)
            )
        except Exception as e:
            content = str(message["text"])

    rprint(Panel(content, 
                title=title, 
                title_align="left", 
                style=Style(color=color)
                ))
    
    documents = [key for key in message.keys() if key != "text"]
    logging.info(f"Document(s): {documents}") if documents else None

async def pretty_stream(agent: "Agent", messages: List[Dict[str, Any]], completion_setting: Dict):
    
    async def stream():
        response = ""
        tokens = 0
        async for chunk in agent.llm.async_gen(
                                messages=messages, 
                                **completion_setting):
            response += chunk
            tokens += get_tokens(agent, chunk)
            yield chunk, tokens, response
    
    if agent.verbose and all_packages_installed:
        console = Console()
        with Live(console=console, refresh_per_second=4) as live:
            async for chunk, tokens, response in stream():
                live.update(
                    Panel(
                        Markdown(str(response)) if agent.markdown_response else str(response),
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
