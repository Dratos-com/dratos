from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
from typing import Any, Dict, Optional
from pydantic import BaseModel
from api.session import Session
from beta.config.config import config


class BaseTool(BaseModel):
    name: str
    description: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Tool must implement __call__ method")

    def execute(
        self, *args: Any, session: Optional[Session] = None, **kwargs: Any
    ) -> Any:
        result = self(*args, **kwargs)
        if session:
            session.data[f"tool_execution_{self.name}"] = {
                "args": args,
                "kwargs": kwargs,
                "result": result,
            }
            config.update_session(session)
        return result
