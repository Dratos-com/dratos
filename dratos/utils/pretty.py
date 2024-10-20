import tiktoken
import json
from typing import List, Dict

from dratos.utils.utils import extract_json_from_str

from rich import print as rprint
from rich.syntax import Syntax
from rich.panel import Panel
from rich.markdown import Markdown
from rich.style import Style
from rich.text import Text
from rich.live import Live
from rich.console import Console, Group

def pretty(agent, message: str, title: str):
    if not agent.verbose:
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
        tokenizer = tiktoken.encoding_for_model(agent.llm.model_name)
        tokens = len(tokenizer.encode(str(message)))

    except KeyError:
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    title = f"{title} ({tokens} tokens)"

    if agent.markdown_response:
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

async def pretty_stream(agent, prompt: str, messages: List[Dict], completion_setting: Dict):
    console = Console()
    try:
        tokenizer = tiktoken.encoding_for_model(agent.llm.model_name)
    except KeyError:
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    async def stream():
        response = ""
        tokens = 0
        async for chunk in agent.llm.async_gen(prompt,
                                messages=agent.messages, 
                                **completion_setting):
            response += chunk
            tokens += len(tokenizer.encode(chunk))
            yield chunk, tokens, response
    
    if agent.verbose:
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
