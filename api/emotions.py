"""
MANORA Emotions API Router.
Provides development and testing endpoints for ad-hoc emotion analysis.
"""

import uuid
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from emotion_agent.emotion_engine import emotion_agent

router = APIRouter(prefix="/emotions", tags=["Emotions"])


class EmotionAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Student input text to analyze")


@router.post("/analyze", status_code=status.HTTP_200_OK)
async def analyze_emotion(payload: EmotionAnalysisRequest) -> Dict[str, Any]:
    """
    Analyzes emotional and behavioral signals in the given text.
    Used for testing, inspection, and ad-hoc analysis.
    """
    try:
        interaction_id = str(uuid.uuid4())
        analysis = await emotion_agent.analyze(
            interaction_id=interaction_id,
            user_id="dev-user",
            session_id="dev-session",
            text=payload.text,
        )
        return analysis.model_dump()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze emotion: {str(e)}",
        )
