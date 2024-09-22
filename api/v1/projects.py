"""
This file contains the API routes for the projects.
"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def get_projects():
    return {"message": "Welcome to the Projects API"}

