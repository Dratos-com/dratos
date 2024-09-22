from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.config import Config
from api.v1 import conversation

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configured Through ENV Vars in .env
config = Config.get_instance().load_config()

ray = config.get_ray(storage_context=config.get_storage, settings=config.get_settings)
daft = config.get_daft(storage_context=config.get_storage, settings=config.get_settings)

@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-Agent Framework API"}

# Include the conversation router with the correct prefix
app.include_router(conversation.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8998)
