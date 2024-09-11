from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
from typing import Dict, Type
from beta.tools.base_tool import BaseTool
from studio.session import Session
from beta.config.config import config


class ToolAccessor:
    _instance = None
    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_tool(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}

    def execute_tool(self, name: str, *args, session: Session = None, **kwargs):
        tool = self.get_tool(name)
        if tool:
            return tool.execute(*args, session=session, **kwargs)
        else:
            raise ValueError(f"Tool '{name}' not found")
