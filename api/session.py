from __future__ import annotations

import typing
if typing.TYPE_CHECKING:
    pass
from datetime import datetime
from typing import Dict, Any
from .base import BaseDBModel
from pydantic import Field


class Session(BaseDBModel):
    __tablename__ = "sessions"
    user_id: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    data: Dict[str, Any] = {}
