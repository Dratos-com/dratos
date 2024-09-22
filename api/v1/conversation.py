from fastapi import APIRouter, Depends, Query
from typing import List, Dict
from beta.agents.agent import Agent

router = APIRouter()


def get_agent():
    # Initialize and return your Agent instance
    return Agent(name="ConversationAgent", memory_db_uri="lancedb")


@router.get("/{conversation_id}/history")
async def get_conversation_history(
    conversation_id: str, agent: Agent = Depends(get_agent)
):
    return await agent.get_conversation_history(conversation_id)


@router.get("/{conversation_id}/branches")
async def get_conversation_branches(
    conversation_id: str, agent: Agent = Depends(get_agent)
):
    return await agent.get_conversation_branches(conversation_id)


@router.post("/{conversation_id}/branch")
async def create_conversation_branch(
    conversation_id: str, branch_name: str, agent: Agent = Depends(get_agent)
):
    return await agent.create_conversation_branch(branch_name)


@router.post("/{conversation_id}/switch-branch")
async def switch_conversation_branch(
    conversation_id: str, branch_name: str, agent: Agent = Depends(get_agent)
):
    return await agent.switch_conversation_branch(branch_name)


@router.post("/{conversation_id}/merge-branches")
async def merge_conversation_branches(
    conversation_id: str,
    source_branch: str,
    target_branch: str,
    agent: Agent = Depends(get_agent),
):
    return await agent.merge_conversation_branches(source_branch, target_branch)


@router.get("/{conversation_id}/page")
async def get_conversation_page(
    conversation_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    agent: Agent = Depends(get_agent),
):
    skip = (page - 1) * page_size
    limit = page_size
    return await agent.get_conversation_page(conversation_id, skip, limit)
