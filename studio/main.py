from __future__ import annotations

import typing
if typing.TYPE_CHECKING:
    pass
from fastapi import FastAPI

from beta.config import Config


app = FastAPI()

# Configured Through ENV Vars in .env
config = Config(is_async=True)

ray = config.get_ray()
uc = config.get_unity_catalog()
daft = config.get_daft()
triton = config.get_triton


# ROUTES ////////////////////////////////////////////////////////
@app.on_event("startup")
async def startup_event():
    plugins = ["plugin1", "plugin2"]  # List of plugin names
    for plugin_name in plugins:
        plugin = load_plugin(plugin_name)
        plugin.Plugin1().initialize()  # Assuming the class is named after the plugin


@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-Agent Framework API"}


# V1 API
from beta.api.v1 import sessions, admin, projects, data, chat

app.include_router(sessions.router, prefix="/api/v1/sessions")
app.include_router(admin.router, prefix="/api/v1/admin")
app.include_router(projects.router, prefix="/api/v1/projects")
app.include_router(artifacts.router, prefix="/api/v1/data")
app.include_router(chats.router, prefix="/api/v1/chat")

# Register your domain objects
register_domain_object(Prompt, PromptAccessor)
# Register other domain objects as needed


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
