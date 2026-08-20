"""
MANORA Chat History API Router.
Exposes per-session chat history for a user with pagination support.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query, status

from database.connection import db

router = APIRouter(prefix="/chat-history", tags=["Chat History"])


@router.get(
    "/{user_id}/sessions/{session_id}",
    status_code=status.HTTP_200_OK,
)
async def get_session_chat_history(
    user_id: str,
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200, description="Max messages to return"),
    offset: int = Query(default=0, ge=0, description="Number of messages to skip"),
) -> Dict[str, Any]:
    """
    Retrieves the full chat history for a student's session.

    Messages are returned in chronological order (oldest first) and include
    both user messages and Buddy responses.
    Supports cursor-based pagination via limit/offset query parameters.
    """
    try:
        messages = await db.get_session_chat_history(
            user_id=user_id,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )

        return {
            "user_id": user_id,
            "session_id": session_id,
            "messages": [
                {
                    "id": str(msg["id"]),
                    "role": msg["role"],
                    "text": msg["raw_text"],
                    "created_at": msg["created_at"].isoformat()
                    if hasattr(msg["created_at"], "isoformat")
                    else str(msg["created_at"]),
                }
                for msg in messages
            ],
            "count": len(messages),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chat history: {str(e)}",
        )
