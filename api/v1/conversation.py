from fastapi import APIRouter, Depends
from typing import List, Dict
from beta.agents.agent import Agent

router = APIRouter()

def get_agent():
    # Initialize and return your Agent instance
    return Agent(name="ConversationAgent")

@router.get("/conversation/{conversation_id}/history")
async def get_conversation_history(conversation_id: str, agent: Agent = Depends(get_agent)):
    return await agent.get_conversation_history(conversation_id)

@router.get("/conversation/{conversation_id}/branches")
async def get_conversation_branches(conversation_id: str, agent: Agent = Depends(get_agent)):
    return await agent.get_conversation_branches(conversation_id)

@router.post("/conversation/{conversation_id}/branch")
async def create_conversation_branch(conversation_id: str, branch_name: str, agent: Agent = Depends(get_agent)):
    return await agent.create_conversation_branch(branch_name)

@router.post("/conversation/{conversation_id}/switch-branch")
async def switch_conversation_branch(conversation_id: str, branch_name: str, agent: Agent = Depends(get_agent)):
    return await agent.switch_conversation_branch(branch_name)

@router.post("/conversation/{conversation_id}/merge-branches")
async def merge_conversation_branches(conversation_id: str, source_branch: str, target_branch: str, agent: Agent = Depends(get_agent)):
    return await agent.merge_conversation_branches(source_branch, target_branch)