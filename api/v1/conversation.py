from fastapi import APIRouter, Depends, Query, HTTPException, Body
from typing import List, Dict
from beta.agents.agent import Agent
from pydantic import BaseModel

router = APIRouter()

def get_agent():
    # Initialize and return your Agent instance
    return Agent(name="ConversationAgent", memory_db_uri="lancedb")

class MessageCreate(BaseModel):
    content: str
    sender: str

class MessageEdit(BaseModel):
    content: str

class BranchCreate(BaseModel):
    branch_name: str
    commit_id: str

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
        return await agent.get_conversation_branches(conversation_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

@router.post("/conversation/{conversation_id}/branch")
async def create_conversation_branch(
    conversation_id: str,
    branch_data: BranchCreate,
    agent: Agent = Depends(get_agent)
):
    try:
        return await agent.create_conversation_branch(conversation_id, branch_data.branch_name, branch_data.commit_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

@router.post("/conversation/{conversation_id}/switch-branch")
async def switch_conversation_branch(
    conversation_id: str, branch_name: str = Body(...), agent: Agent = Depends(get_agent)
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

@router.post("/conversation/{conversation_id}/message")
async def send_message(
    conversation_id: str,
    message: MessageCreate,
    agent: Agent = Depends(get_agent)
):
    try:
        return await agent.add_message(conversation_id, message.content, message.sender)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/conversation/{conversation_id}/message/{message_id}")
async def edit_message(
    conversation_id: str,
    message_id: str,
    message: MessageEdit,
    agent: Agent = Depends(get_agent)
):
    try:
        return await agent.edit_message(conversation_id, message_id, message.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
