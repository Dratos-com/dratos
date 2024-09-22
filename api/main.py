from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    pass
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.config import Config


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


# ROUTES ////////////////////////////////////////////////////////
# @app.on_event("startup")
# async def startup_event():
#     plugins = ["plugin1", "plugin2"]  # List of plugin names
#     for plugin_name in plugins:
#         plugin = load_plugin(plugin_name)
#         plugin.Plugin1().initialize()  # Assuming the class is named after the plugin


@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-Agent Framework API"}


# V1 API
# from v1 import sessions, admin, projects, data, chat, conversation
from api.v1 import conversation

# app.include_router(sessions.router, prefix="/api/v1/sessions")
# app.include_router(admin.router, prefix="/api/v1/admin")
# app.include_router(projects.router, prefix="/api/v1/projects")
# app.include_router(data.router, prefix="/api/v1/data")
# app.include_router(chat.router, prefix="/api/v1/chat")
app.include_router(conversation.router, prefix="/api/v1/conversation")

# Register your domain objects
# register_domain_object(Prompt, PromptAccessor)
# Register other domain objects as needed


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8998)
