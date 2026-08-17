"""MANORA Data Agent schema definitions.

``CandidateMemory`` remains the evidence/input contract produced by the
extraction stage.  The remaining models describe the isolated V1
consolidation result and deliberately contain no persistence concerns.
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
    # Optional for backwards compatibility with the existing extraction path.
    # DataAgent.consolidate() requires this value so evidence is never compared
    # across users.
    user_id: Optional[str] = None
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


class PromotedMemory(BaseModel):
    """Durable knowledge consolidated from one or more candidate memories."""

    id: str
    user_id: str
    content: str
    topic: str
    evidence_ids: List[str] = Field(default_factory=list)
    emotional_state: List[CandidateMemoryEmotion] = Field(default_factory=list)
    contexts: List[Dict[str, Any]] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    support_count: int = Field(default=1, ge=1)
    importance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class Pattern(BaseModel):
    """A repeated theme supported by multiple independent observations."""

    id: str
    user_id: str
    name: str
    description: str
    pattern_type: str
    topic: str
    evidence_ids: List[str] = Field(default_factory=list)
    promoted_memory_ids: List[str] = Field(default_factory=list)
    occurrence_count: int = Field(ge=2)
    emotional_state: List[CandidateMemoryEmotion] = Field(default_factory=list)
    importance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class Relationship(BaseModel):
    """A deterministic semantic relationship between consolidated entities."""

    id: str
    user_id: str
    source_id: str
    source_type: str
    relation: str
    target_id: str
    target_type: str
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class DataAgentResult(BaseModel):
    """Complete, side-effect-free output of Data Agent V1 consolidation."""

    promoted_memories: List[PromotedMemory] = Field(default_factory=list)
    patterns: List[Pattern] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    rejected_memory_ids: List[str] = Field(default_factory=list)
