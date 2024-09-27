from typing import Callable, Any
import re

def prompt(func: Callable[..., str]) -> Callable[..., str]:
    """
    A decorator that processes a function's docstring as a template.
    
    This decorator replaces placeholders in the function's docstring
    with the values of the function's arguments.
    
    Args:
        func (Callable[..., str]): The function to be decorated.
    
    Returns:
        Callable[..., str]: The wrapped function that returns the processed docstring.
    """
    def wrapper(*args: Any, **kwargs: Any) -> str:
        # Get the function's docstring
        template = func.__doc__
        if not template:
            raise ValueError(f"Function {func.__name__} has no docstring to use as a template.")
        
        # Get the function's argument names
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        
        # Combine positional and keyword arguments
        all_args = dict(zip(arg_names, args))
        all_args.update(kwargs)
        
        # Replace placeholders in the template
        for key, value in all_args.items():
            placeholder = f"{{{{{key}}}}}"
            template = template.replace(placeholder, str(value))
        
        # Remove any remaining placeholders
        template = re.sub(r'\{\{.*?\}\}', '', template)
        
        return template.strip()
    
    return wrapper