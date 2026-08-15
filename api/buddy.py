"""
MANORA Buddy State API Router.
Provides endpoints to inspect Buddy's internal emotional state and transition history.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, status

from database.connection import db
from state.state_engine import state_engine

router = APIRouter(prefix="/buddy", tags=["Buddy"])


@router.get("/state/{user_id}", status_code=status.HTTP_200_OK)
async def get_user_buddy_state(user_id: str) -> Dict[str, Any]:
    """
    Retrieves the current internal emotional state of Buddy for a student.
    Returns baseline state if no interactions have occurred yet.
    """
    try:
        raw_state = await db.get_buddy_state(user_id=user_id)
        if raw_state:
            return {
                "user_id": user_id,
                "state": {
                    "happiness": raw_state.get("happiness"),
                    "sadness": raw_state.get("sadness"),
                    "frustration": raw_state.get("frustration"),
                    "concern": raw_state.get("concern"),
                    "warmth": raw_state.get("warmth"),
                    "patience": raw_state.get("patience"),
                    "energy": raw_state.get("energy"),
                },
                "updated_at": raw_state.get("updated_at"),
            }
        # Baseline fallback
        baseline = state_engine.initialize_state()
        return {
            "user_id": user_id,
            "state": baseline.to_dict(),
            "updated_at": None,
            "is_initial": True,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Buddy state: {str(e)}",
        )


@router.get("/state/{user_id}/history", status_code=status.HTTP_200_OK)
async def get_user_buddy_state_history(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Retrieves the chronological audit history of Buddy state transitions for a student.
    """
    try:
        history = await db.get_buddy_state_history(user_id=user_id, limit=limit)
        return history
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Buddy state history: {str(e)}",
        )
