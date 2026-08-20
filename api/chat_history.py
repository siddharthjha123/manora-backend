"""
MANORA Chat History API Router.
Exposes user sessions and per-session chat history with pagination support.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query, status

from database.connection import db

router = APIRouter(prefix="/chat-history", tags=["Chat History"])


@router.get(
    "/{user_id}/sessions",
    status_code=status.HTTP_200_OK,
)
async def get_user_sessions(
    user_id: str,
    active_only: bool = Query(default=False, description="Return only active sessions"),
    limit: int = Query(default=20, ge=1, le=100, description="Max sessions to return"),
    offset: int = Query(default=0, ge=0, description="Number of sessions to skip"),
) -> Dict[str, Any]:
    """
    Retrieves all conversation sessions for a student.

    Sessions are returned in reverse chronological order (most recent first).
    Use `active_only=true` to filter to currently active sessions.
    """
    try:
        sessions = await db.get_user_sessions(
            user_id=user_id,
            active_only=active_only,
            limit=limit,
            offset=offset,
        )

        return {
            "user_id": user_id,
            "sessions": [
                {
                    "id": str(s["id"]),
                    "title": s.get("title"),
                    "is_active": s.get("is_active", True),
                    "created_at": s["created_at"].isoformat()
                    if hasattr(s["created_at"], "isoformat")
                    else str(s["created_at"]),
                    "updated_at": s["updated_at"].isoformat()
                    if hasattr(s["updated_at"], "isoformat")
                    else str(s["updated_at"]),
                }
                for s in sessions
            ],
            "count": len(sessions),
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sessions: {str(e)}",
        )


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
