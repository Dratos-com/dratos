from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
from beta.tools.base_tool import BaseTool


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Performs basic arithmetic operations"

    def __call__(self, operation: str, a: float, b: float) -> float:
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")


# Register the tool
config.register_tool(CalculatorTool())
