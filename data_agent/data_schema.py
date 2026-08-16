"""
MANORA Data Agent - Schema Definitions.
Defines contracts for candidate memories extracted from interactions.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CandidateMemoryEvent(BaseModel):
    type: str = Field(..., description="Event classification (e.g. study_avoidance, exam_preparation)")
    description: str = Field(..., description="Details of the event occurred")


class CandidateMemoryBehavior(BaseModel):
    type: str = Field(..., description="Behavioral classification (e.g. avoidance, procrastination, engagement)")
    description: str = Field(..., description="Description of the student's behavior")


class CandidateMemoryDecision(BaseModel):
    description: str = Field(..., description="Description of the decision or choice made by student")


class CandidateMemoryGoalRelevance(BaseModel):
    related: bool = Field(default=False)
    goal: Optional[str] = Field(default=None)


class CandidateMemoryEmotion(BaseModel):
    emotion: str
    confidence: float = Field(ge=0.0, le=1.0)


class CandidateMemory(BaseModel):
    """Structured candidate memory produced by Data Agent for storage evaluation."""
    id: Optional[str] = None
    content: str = Field(..., description="Core narrative content of the memory")
    context: Dict[str, Any] = Field(default_factory=dict, description="Topic, subtopic metadata")
    emotional_state: List[CandidateMemoryEmotion] = Field(default_factory=list)
    events: List[CandidateMemoryEvent] = Field(default_factory=list)
    behavior: Optional[CandidateMemoryBehavior] = None
    decision: Optional[CandidateMemoryDecision] = None
    goal_relevance: Optional[CandidateMemoryGoalRelevance] = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Memory importance score")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Extraction confidence score")

    @field_validator("importance", "confidence", mode="before")
    @classmethod
    def clamp_scores(cls, v: Any) -> float:
        try:
            val = float(v)
            return round(max(0.0, min(1.0, val)), 3)
        except (ValueError, TypeError):
            return 0.5


class DataAgentOutput(BaseModel):
    """Output envelope from Data Agent."""
    candidate_memories: List[CandidateMemory] = Field(default_factory=list)
