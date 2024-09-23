from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from lancedb_store import LanceDBStore
import os

app = FastAPI()

# Initialize LanceDBStore
db_path = os.environ.get("LANCEDB_PATH", "/path/to/your/lancedb")
store = LanceDBStore(db_path)

class Post(BaseModel):
    title: str
    content: str
    author: str
    timePoint: str
    tags: str

@app.post("/api/v1/posts")
async def create_post(post: Post):
    # ... existing code for creating a post ...

@app.get("/api/v1/posts")
async def get_posts():
    try:
        posts = store.get_all_posts()
        return posts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/debug/posts")
async def debug_posts():
    try:
        posts = store.get_all_posts()
        return {
            "post_count": len(posts),
            "first_post": posts[0] if posts else None,
            "db_path": db_path
        }
    except Exception as e:
        return {"error": str(e)}

# ... other endpoints ...