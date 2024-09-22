from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Dict
from beta.agents.agent import Agent

router = APIRouter()


def get_agent():
    # Initialize and return your Agent instance
    return Agent(name="ConversationAgent", memory_db_uri="lancedb")


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
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/conversation/{conversation_id}/branch")
async def create_conversation_branch(
    conversation_id: str, branch_name: str, agent: Agent = Depends(get_agent)
):
    try:
        return await agent.create_conversation_branch(branch_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversation/{conversation_id}/switch-branch")
async def switch_conversation_branch(
    conversation_id: str, branch_name: str, agent: Agent = Depends(get_agent)
):
    try:
        return await agent.switch_conversation_branch(branch_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/conversation/{conversation_id}/merge-branches")
async def merge_conversation_branches(
    conversation_id: str,
    source_branch: str,
    target_branch: str,
    agent: Agent = Depends(get_agent),
):
    try:
        return await agent.merge_conversation_branches(source_branch, target_branch)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
        raise HTTPException(status_code=404, detail=str(e))
