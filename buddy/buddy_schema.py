"""
MANORA Buddy Agent - Schemas and Data Contracts.
Defines structured response format, expression enums, and response types for the Buddy Agent.
"""

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class BuddyExpression(str, Enum):
    """Visual/emotional expression states supported for frontend avatar rendering."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    CONCERNED = "concerned"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    ENCOURAGING = "encouraging"
    THOUGHTFUL = "thoughtful"


class BuddyResponseType(str, Enum):
    """Categorization of Buddy's conversational stance."""
    REFLECTION = "reflection"
    QUESTION = "question"
    CHALLENGE = "challenge"
    VALIDATION = "validation"
    GUIDANCE = "guidance"
    SUPPORT = "support"


class BuddyResponse(BaseModel):
    """Complete structured response emitted by the Buddy Agent."""
    text: str = Field(..., min_length=1, description="Natural language response text from Buddy")
    expression: str = Field(
        default=BuddyExpression.NEUTRAL.value,
        description="Emotional facial expression for frontend rendering"
    )
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Expression intensity score between 0.0 and 1.0"
    )
    response_type: str = Field(
        default=BuddyResponseType.SUPPORT.value,
        description="Stance/type of the response"
    )

    @field_validator("intensity", mode="before")
    @classmethod
    def clamp_intensity(cls, v: Any) -> float:
        try:
            val = float(v)
            return round(max(0.0, min(1.0, val)), 3)
        except (ValueError, TypeError):
            return 0.5

    @field_validator("expression", mode="before")
    @classmethod
    def normalize_expression(cls, v: Any) -> str:
        if not v:
            return BuddyExpression.NEUTRAL.value
        val_str = str(v).lower().strip()
        valid_expressions = {e.value for e in BuddyExpression}
        if val_str in valid_expressions:
            return val_str
        # Map nearest synonyms if LLM outputs variation
        mapping = {
            "worried": "concerned",
            "empathetic": "thoughtful",
            "curious": "thoughtful",
            "irritated": "frustrated",
            "cheerful": "happy",
            "caring": "encouraging",
            "warm": "encouraging",
        }
        return mapping.get(val_str, BuddyExpression.NEUTRAL.value)

    @field_validator("response_type", mode="before")
    @classmethod
    def normalize_response_type(cls, v: Any) -> str:
        if not v:
            return BuddyResponseType.SUPPORT.value
        val_str = str(v).lower().strip()
        valid_types = {t.value for t in BuddyResponseType}
        if val_str in valid_types:
            return val_str
        return BuddyResponseType.REFLECTION.value
