from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
from beta.config.config import config
from studio.session import Session


class SessionManager:
    @staticmethod
    def create_session(user_id: str) -> Session:
        return config.create_session(user_id)

    @staticmethod
    def update_session(session: Session):
        config.update_session(session)

    @staticmethod
    def autosave_session(session: Session):
        # This would typically be called periodically or on certain events
        worker_url = "https://your-autosave-worker.workers.dev"
        response = httpx.post(worker_url, json=session.dict())
        if response.status_code == 200:
            print("Session autosaved successfully")
        else:
            print(f"Autosave failed: {response.text}")
