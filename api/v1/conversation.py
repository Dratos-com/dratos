import os
from fastapi import FastAPI, APIRouter, Depends, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
from beta.agents.agent import (
    Agent,
)  # Ensure this module is correctly installed and accessible
from pydantic import BaseModel, Field
import logging
from datetime import datetime
from dotenv import load_dotenv
from beta.models.serve.engines.openai_engine import OpenAIEngine, OpenAIEngineConfig
import traceback
from pydantic import ValidationError
import numpy as np

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

router = APIRouter()

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]


def get_agent():
    # Initialize and return your Agent instance
    oai_config = OpenAIEngineConfig(
        data={
            "api_key": api_key,
            "base_url": "https://api.openai.com/v1",
            "model_name": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
    )

    # Initialize the OpenAI engine
    openai_engine = OpenAIEngine(config=oai_config)

    # Ensure the paths are correct
    memory_db_uri = "./memory/lancedb"
    git_repo_path = "./memory/memories_repo"

    # Initialize the Agent with the correct paths
    agent = Agent(
        name="ConversationAgent",
        memory_db_uri=memory_db_uri,
        git_repo_path=git_repo_path,
        is_async=True,
        engine=openai_engine,
    )
    return agent


class MessageCreate(BaseModel):
    content: str
    sender: str


class MessageEdit(BaseModel):
    content: str


class BranchResponse(BaseModel):
    branch_name: str
    commit_id: str
    history: List[Dict]


class BranchCreate(BaseModel):
    new_branch_id: str
    commit_id: str


class HistoryItem(BaseModel):
    id: str
    content: str
    sender: str
    timestamp: str

    @classmethod
    def from_dict(cls, data: Dict):
        # Convert timestamp to string if it's a datetime object
        if isinstance(data.get("timestamp"), datetime):
            data["timestamp"] = data["timestamp"].isoformat()
        return cls(**data)


class MessageResponse(BaseModel):
    id: str
    content: str
    sender: str
    timestamp: str
    commit_id: str
    history: List[HistoryItem]


class PostCreate(BaseModel):
    title: str
    content: str
    author: str
    timePoint: str
    tags: str
    parent_id: Optional[str] = None
    version: float = Field(default=1.0, description="Version of the post as a float32")


class CommentCreate(BaseModel):
    content: str
    author: str


class Vote(BaseModel):
    type: str  # 'up' or 'down'


class Post(BaseModel):
    id: str
    title: Optional[str]
    content: Optional[str]
    author: str
    timestamp: str
    upvotes: int = 0
    downvotes: int = 0
    comments: int = 0
    branches: int = 0
    timePoint: Optional[str] = ""
    tags: Optional[str] = ""
    parent_id: Optional[str] = None
    version: float = Field(default=1.0, description="Version of the post as a float32")

    class Config:
        json_encoders = {
            np.float32: lambda v: float(v)  # Convert numpy.float32 to Python float
        }


@router.post("/conversation/{conversation_id}/branch", response_model=BranchResponse)
async def create_conversation_branch(
    conversation_id: str, branch_data: BranchCreate, agent: Agent = Depends(get_agent)
):
    try:
        # Retrieve the latest commit ID for the conversation
        latest_commit = await agent.get_latest_commit_id(conversation_id)
        result = await agent.create_conversation_branch(
            conversation_id, branch_data.new_branch_id, latest_commit
        )
        return BranchResponse(
            branch_name=result["branch_name"],
            commit_id=result["commit_id"],
            history=result["history"],
        )
    except Exception as e:
        print(f"Error in create_conversation_branch: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/conversation/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str, agent: Agent = Depends(get_agent)
):
    try:
        return await agent.get_conversation_history(conversation_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/conversation/{conversation_id}/branches")
async def get_conversation_branches(
    conversation_id: str, agent: Agent = Depends(get_agent)
):
    try:
        branches = await agent.get_conversation_branches(conversation_id)
        return branches
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/conversation/{conversation_id}/switch-branch")
async def switch_conversation_branch(
    conversation_id: str,
    branch_name: str = Body(...),
    agent: Agent = Depends(get_agent),
):
    try:
        return await agent.switch_conversation_branch(branch_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/conversation/{conversation_id}/merge-branches")
async def merge_conversation_branches(
    conversation_id: str,
    source_branch: str = Body(...),
    target_branch: str = Body(...),
    agent: Agent = Depends(get_agent),
):
    try:
        return await agent.merge_conversation_branches(source_branch, target_branch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/conversation/{conversation_id}/page")
async def get_conversation_page(
    conversation_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    agent: Agent = Depends(get_agent),
):
    try:
        skip = (page - 1) * page_size
        limit = page_size
        return await agent.get_conversation_page(conversation_id, skip, limit)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/conversation/{conversation_id}/message", response_model=MessageResponse)
async def post_message(
    conversation_id: str, message: MessageCreate, agent: Agent = Depends(get_agent)
):
    try:
        new_message = await agent.add_message(
            conversation_id, message.content, message.sender
        )
        logger.debug(f"New message to return: {new_message}")

        # Ensure history is a list of HistoryItem
        history = [
            HistoryItem.from_dict(item) for item in new_message.get("history", [])
        ]

        # Create and return a MessageResponse instance
        return MessageResponse(
            id=new_message["id"],
            content=new_message["content"],
            sender=new_message["sender"],
            timestamp=new_message["timestamp"],
            commit_id=new_message["commit_id"],
            history=history,
        )
    except Exception as e:
        logger.error(f"Error in post_message: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.put("/conversation/{conversation_id}/message/{message_id}")
async def edit_message(
    conversation_id: str,
    message_id: str,
    message: MessageEdit,
    agent: Agent = Depends(get_agent),
):
    try:
        edited_message = await agent.edit_message(
            conversation_id, message_id, message.content
        )
        return MessageResponse(
            id=edited_message["message_id"],
            content=edited_message["content"],
            sender=edited_message["role"],
            timestamp=edited_message["timestamp"],
            commit_id=edited_message["commit_id"],
            history=[],  # We're not handling history in this response
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post(
    "/conversation/{conversation_id}/send-message", response_model=MessageResponse
)
async def send_message(
    conversation_id: str, message: MessageCreate, agent: Agent = Depends(get_agent)
):
    try:
        agent_message = await agent.process_message(
            conversation_id, message.content, message.sender
        )
        return MessageResponse(
            id=agent_message["id"],
            content=agent_message["content"],
            sender=agent_message["sender"],
            timestamp=agent_message["timestamp"],
            commit_id=agent_message["commit_id"],
            history=[],  # We're not handling history in this response
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


class PostResponse(BaseModel):
    user_post: Post
    agent_response: Optional[Post]


@router.post("/posts", response_model=PostResponse)
async def create_post(post: PostCreate, agent: Agent = Depends(get_agent)):
    try:
        print(f"Received post data: {post}")
        new_post, agent_response = await agent.create_post(
            post.title,
            post.content,
            post.author,
            post.timePoint,
            post.tags,
            post.parent_id,
            np.float32(post.version),
        )

        user_post_response = Post(
            id=new_post["id"],
            title=new_post["title"],
            content=new_post["content"],
            author=new_post["author"],
            timestamp=new_post["timestamp"] if isinstance(new_post["timestamp"], str) else new_post["timestamp"].isoformat(),
            upvotes=new_post["upvotes"],
            downvotes=new_post["downvotes"],
            comments=len(new_post["comments"]),
            branches=len(new_post["branches"]),
            timePoint=new_post["timePoint"],
            tags=new_post["tags"],
            parent_id=new_post.get("parent_id"),
            version=np.float32(new_post.get("version", 1.0)),
        )

        agent_post_response = Post(
            id=agent_response["id"],
            title=agent_response["title"],
            content=agent_response["content"],
            author=agent_response["author"],
            timestamp=agent_response["timestamp"] if isinstance(agent_response["timestamp"], str) else agent_response["timestamp"].isoformat(),
            upvotes=agent_response["upvotes"],
            downvotes=agent_response["downvotes"],
            comments=len(agent_response["comments"]),
            branches=len(agent_response["branches"]),
            timePoint=agent_response["timePoint"],
            tags=agent_response["tags"],
            parent_id=agent_response.get("parent_id"),
            version=np.float32(agent_response.get("version", 1.0)),
        )

        print(f"Created new post: {user_post_response}")
        print(f"Created agent response: {agent_post_response}")

        return PostResponse(
            user_post=user_post_response, agent_response=agent_post_response
        )
    except Exception as e:
        print(f"Error creating post: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/posts", response_model=List[Post])
async def get_posts(agent: Agent = Depends(get_agent)):
    try:
        raw_posts = await agent.get_all_posts()
        print(f"Raw posts: {raw_posts}")

        posts = []
        for raw_post in raw_posts:
            try:
                post = Post(
                    id=raw_post["id"],
                    title=raw_post.get("title"),
                    content=raw_post.get("content"),
                    author=raw_post["author"],
                    timestamp=raw_post["timestamp"] if isinstance(raw_post["timestamp"], str) else raw_post["timestamp"].isoformat(),
                    upvotes=raw_post.get("upvotes", 0),
                    downvotes=raw_post.get("downvotes", 0),
                    comments=len(raw_post.get("comments", [])),
                    branches=len(raw_post.get("branches", [])),
                    timePoint=raw_post.get("timePoint", ""),
                    tags=raw_post.get("tags", ""),
                    parent_id=raw_post.get("parent_id"),
                    version=np.float32(raw_post.get("version", 1.0)),
                )
                posts.append(post)
            except ValidationError as ve:
                print(
                    f"Validation error for post {raw_post.get('id', 'unknown')}: {ve}"
                )
            except Exception as e:
                print(
                    f"Error processing post {raw_post.get('id', 'unknown')}: {str(e)}"
                )

        return posts
    except Exception as e:
        print(f"Error fetching posts: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching posts: {str(e)}"
        ) from e


@router.get("/posts/{post_id}")
async def get_post(post_id: str, agent: Agent = Depends(get_agent)):
    try:
        return await agent.get_post(post_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post("/posts/{post_id}/comments")
async def add_comment(
    post_id: str, comment: CommentCreate, agent: Agent = Depends(get_agent)
):
    try:
        return await agent.add_comment(post_id, comment.content, comment.author)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/posts/{post_id}/vote")
async def vote_post(post_id: str, vote: Vote, agent: Agent = Depends(get_agent)):
    try:
        return await agent.vote_post(post_id, vote.type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/posts/{post_id}/branches")
async def create_post_branch(
    post_id: str, parent_id: str, agent: Agent = Depends(get_agent)
):
    try:
        return await agent.create_post_branch(post_id, parent_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


app.include_router(router, prefix="/api/v1")
