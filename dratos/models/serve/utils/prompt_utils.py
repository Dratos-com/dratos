import functools
import inspect
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Dict

from jinja2 import Environment, StrictUndefined


@dataclass
class Prompt:
    """Represents a prompt function."""

    template: str
    signature: inspect.Signature

    def __post_init__(self):
        self.parameters: List[str] = list(self.signature.parameters.keys())

    def __call__(self, *args, **kwargs) -> str:
        """Render and return the template."""
        bound_arguments = self.signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        return render(self.template, **bound_arguments.arguments)

    def __str__(self):
        return self.template


def prompt(fn: Callable) -> Prompt:
    """Decorate a function that contains a prompt template."""
    signature = inspect.signature(fn)

    docstring = fn.__doc__
    if docstring is None:
        raise TypeError("Could not find a template in the function's docstring.")

    template = docstring

    return Prompt(template, signature)


def render(template: str, **values: Optional[Dict[str, Any]]) -> str:
    """Parse a Jinja2 template and render it."""
    cleaned_template = inspect.cleandoc(template)

    ends_with_linebreak = template.replace(" ", "").endswith("\n\n")
    if ends_with_linebreak:
        cleaned_template += "\n"

    cleaned_template = re.sub(r"(?![\r\n])(\b\s+)", " ", cleaned_template)

    env = Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )
    jinja_template = env.from_string(cleaned_template)

    return jinja_template.render(**values)
