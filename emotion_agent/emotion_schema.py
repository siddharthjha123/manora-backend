"""
MANORA Emotion Agent - Schemas and Pydantic Models.
Defines structured data contracts for emotion analysis and validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class EmotionItem(BaseModel):
    """Represents a specific detected emotion, its intensity, confidence, and source."""
    emotion: str = Field(..., description="Name of the emotion (e.g., frustration, guilt, anxiety, joy)")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Intensity score between 0.0 and 1.0")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    source: str = Field(
        default="model_inferred",
        description="Source of detection: model_inferred, ml_classifier, or user_stated"
    )

    @field_validator("intensity", "confidence", mode="before")
    @classmethod
    def clamp_bounds(cls, v):
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.5


class GoalRelevance(BaseModel):
    """Indicates if and how the interaction relates to student's academic or personal goals."""
    related: bool = Field(default=False, description="Whether interaction relates to known goals")
    goal: Optional[str] = Field(default=None, description="Name or summary of the related goal")


class EmotionAnalysis(BaseModel):
    """Complete structured output produced by the Emotion Agent."""
    interaction_id: str = Field(..., description="UUID of the interaction analyzed")
    primary_emotion: str = Field(..., description="The predominant emotion detected")
    emotions: List[EmotionItem] = Field(default_factory=list, description="List of granular emotions detected")
    emotional_summary: str = Field(..., description="Concise qualitative summary of the student's emotional state")
    behavioral_signals: List[str] = Field(
        default_factory=list,
        description="Behavioral patterns or actions observed (e.g. avoided study activity)"
    )
    decision_signals: List[str] = Field(
        default_factory=list,
        description="Decisions made by the student (e.g. chose entertainment over studying)"
    )
    goal_relevance: GoalRelevance = Field(
        default_factory=GoalRelevance,
        description="Goal connection metadata"
    )


class EmotionAnalysisRequest(BaseModel):
    """API request schema for ad-hoc / dev emotion analysis."""
    text: str = Field(..., min_length=1, description="Student input text to analyze")
    interaction_id: Optional[str] = None
    user_id: Optional[str] = "dev-user"
    session_id: Optional[str] = "dev-session"
