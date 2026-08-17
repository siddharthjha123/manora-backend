"""Pydantic contracts for candidate evidence and Data Agent V1 results."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CandidateMemoryEvent(BaseModel):
    type: str = Field(..., description="Event classification")
    description: str = Field(..., description="Details of the event")


class CandidateMemoryBehavior(BaseModel):
    type: str = Field(..., description="Behavioral classification")
    description: str = Field(..., description="Description of the behavior")


class CandidateMemoryDecision(BaseModel):
    description: str = Field(..., description="Decision or choice made by the student")


class CandidateMemoryGoalRelevance(BaseModel):
    related: bool = False
    goal: Optional[str] = None


class CandidateMemoryEmotion(BaseModel):
    emotion: str
    confidence: float = Field(ge=0.0, le=1.0)


class CandidateMemory(BaseModel):
    """Evidence produced by the extraction stage; not long-term memory."""

    id: Optional[str] = None
    user_id: Optional[str] = None
    content: str = Field(..., description="Core narrative content of the observation")
    context: Dict[str, Any] = Field(default_factory=dict)
    emotional_state: List[CandidateMemoryEmotion] = Field(default_factory=list)
    events: List[CandidateMemoryEvent] = Field(default_factory=list)
    behavior: Optional[CandidateMemoryBehavior] = None
    decision: Optional[CandidateMemoryDecision] = None
    goal_relevance: Optional[CandidateMemoryGoalRelevance] = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("importance", "confidence", mode="before")
    @classmethod
    def clamp_scores(cls, value: Any) -> float:
        try:
            score = float(value)
            return round(max(0.0, min(1.0, score)), 3)
        except (ValueError, TypeError):
            return 0.5


class DataAgentOutput(BaseModel):
    """Legacy extraction output retained for the existing parser."""

    candidate_memories: List[CandidateMemory] = Field(default_factory=list)


class ExistingLongTermMemory(BaseModel):
    """Persistent memory supplied as context for possible UPDATE actions."""

    id: str
    user_id: str
    content: str
    evidence_ids: List[str] = Field(default_factory=list)
    emotions: List[CandidateMemoryEmotion] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryActionType(str, Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    REJECT = "REJECT"


class RelationshipType(str, Enum):
    INFLUENCES = "INFLUENCES"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    CONTRIBUTES_TO = "CONTRIBUTES_TO"
    TRIGGERS = "TRIGGERS"
    RELATED_TO = "RELATED_TO"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"


class RelationshipEntityType(str, Enum):
    MEMORY_ACTION = "memory_action"
    LONG_TERM_MEMORY = "long_term_memory"
    CANDIDATE_MEMORY = "candidate_memory"
    PATTERN = "pattern"
    GOAL = "goal"


class MemoryAction(BaseModel):
    """One LLM decision about candidate evidence."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(..., min_length=1, description="Temporary logical ID")
    action: MemoryActionType
    memory_id: Optional[str] = None
    candidate_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    content: Optional[str] = None
    emotions: List[CandidateMemoryEmotion] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_action_shape(self):
        if self.action == MemoryActionType.UPDATE and not self.memory_id:
            raise ValueError("UPDATE requires memory_id")
        if self.action != MemoryActionType.UPDATE and self.memory_id is not None:
            raise ValueError("Only UPDATE may reference an existing memory_id")
        if self.action in {
            MemoryActionType.CREATE,
            MemoryActionType.UPDATE,
            MemoryActionType.MERGE,
        } and not (self.content and self.content.strip()):
            raise ValueError(f"{self.action.value} requires consolidated content")
        if self.action == MemoryActionType.MERGE and len(self.candidate_ids) < 2:
            raise ValueError("MERGE requires at least two candidate_ids")
        return self


class Relationship(BaseModel):
    """Evidence-backed relationship discovered by the memory reasoning agent."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1)
    source_type: RelationshipEntityType
    relation: RelationshipType
    target_id: str = Field(..., min_length=1)
    target_type: RelationshipEntityType
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class DataAgentResult(BaseModel):
    """Validated, side-effect-free output of Data Agent V1."""

    model_config = ConfigDict(extra="forbid")

    user_id: Optional[str] = None
    memory_actions: List[MemoryAction] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    reasoning_summary: str = "No candidate memories were supplied."
