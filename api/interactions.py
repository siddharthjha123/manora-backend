"""
MANORA Interactions API Router.
Handles POST /interactions endpoint for student conversational turns.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.interaction_service import interaction_service

router = APIRouter(prefix="/interactions", tags=["Interactions"])


class InteractionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Student user identifier (UUID or string)")
    session_id: str = Field(..., min_length=1, description="Active session identifier (UUID or string)")
    text: str = Field(..., min_length=1, description="Student message text")


@router.post("", status_code=status.HTTP_200_OK)
async def create_interaction(payload: InteractionRequest) -> Dict[str, Any]:
    """
    Processes student interaction through the full MANORA pipeline:
    1. Records interaction
    2. Decides memory retrieval
    3. Analyzes emotions & behavioral signals
    4. Evaluates candidate memories
    5. Updates Buddy internal state deterministically
    6. Generates Buddy response with expression & stance
    """
    try:
        response = await interaction_service.process_interaction(
            user_id=payload.user_id,
            session_id=payload.session_id,
            text=payload.text,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process interaction: {str(e)}",
        )
