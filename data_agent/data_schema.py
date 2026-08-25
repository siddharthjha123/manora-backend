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


class RelationshipType(str, Enum):
    INFLUENCES = "INFLUENCES"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    CONTRIBUTES_TO = "CONTRIBUTES_TO"
    TRIGGERS = "TRIGGERS"
    RELATED_TO = "RELATED_TO"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"


class ConsolidationRelationship(BaseModel):
    """Evidence-backed relationship between two Stage 1 consolidated groups."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1)
    relation: RelationshipType
    target_id: str = Field(..., min_length=1)
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ConsolidatedMemory(BaseModel):
    """Stage 1 result: candidate evidence combined into one memory statement."""

    model_config = ConfigDict(extra="forbid")

    consolidation_id: str = Field(..., min_length=1)
    candidate_ids: List[str] = Field(..., min_length=1)
    evidence_ids: List[str] = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    emotions: List[CandidateMemoryEmotion] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Stage1ConsolidationResult(BaseModel):
    """Complete output of candidate-only Stage 1 LLM reasoning."""

    model_config = ConfigDict(extra="forbid")

    user_id: Optional[str] = None
    consolidated_memories: List[ConsolidatedMemory] = Field(default_factory=list)
    rejected_candidate_ids: List[str] = Field(default_factory=list)
    relationships: List[ConsolidationRelationship] = Field(default_factory=list)
    reasoning_summary: str = "No candidate memories were supplied."


class MemoryDecisionType(str, Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    REJECT = "REJECT"


class MemoryDecision(BaseModel):
    """Stage 2 decision for one consolidated memory."""

    model_config = ConfigDict(extra="forbid")

    action: MemoryDecisionType
    memory_id: Optional[str] = None
    candidate_ids: List[str] = Field(..., min_length=1)
    evidence_ids: List[str] = Field(..., min_length=1)
    content: Optional[str] = None
    emotions: List[CandidateMemoryEmotion] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_action_shape(self):
        if self.action == MemoryDecisionType.UPDATE and not self.memory_id:
            raise ValueError("UPDATE requires memory_id")
        if self.action != MemoryDecisionType.UPDATE and self.memory_id is not None:
            raise ValueError("Only UPDATE may reference an existing memory_id")
        if self.action in {MemoryDecisionType.CREATE, MemoryDecisionType.UPDATE}:
            if not (self.content and self.content.strip()):
                raise ValueError(f"{self.action.value} requires content")
        if self.action == MemoryDecisionType.REJECT and self.content is not None:
            raise ValueError("REJECT content must be null")
        return self


class Stage2MemoryDecisionResult(BaseModel):
    """Validated Stage 2 LLM response for one consolidated memory."""

    model_config = ConfigDict(extra="forbid")

    user_id: str
    decision: MemoryDecision


class RetrievalResult(BaseModel):
    """Qdrant candidates considered for one Stage 2 decision."""

    consolidation_id: str
    query: str
    existing_memories: List[Dict[str, Any]] = Field(default_factory=list)


class PersistenceResult(BaseModel):
    """Observable persistence outcome for one Stage 2 decision."""

    action: MemoryDecisionType
    memory_id: Optional[str] = None
    postgres_operation: str
    qdrant_operation: str
    memory: Optional[Dict[str, Any]] = None


class GraphRelationshipStatus(str, Enum):
    """Outcome of translating and persisting one Stage 1 relationship."""

    CREATED = "CREATED"
    SKIPPED_MISSING_ENDPOINT = "SKIPPED_MISSING_ENDPOINT"
    REJECTED = "REJECTED"


class GraphRelationshipResult(BaseModel):
    """Observable Stage-1-group to final-memory relationship translation."""

    source_consolidation_id: str
    relation: RelationshipType
    target_consolidation_id: str
    source_memory_id: Optional[str] = None
    target_memory_id: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    status: GraphRelationshipStatus


class MemoryPipelineResult(BaseModel):
    """Complete two-stage pipeline result returned by MemoryEngine."""

    stage1: Stage1ConsolidationResult
    retrievals: List[RetrievalResult] = Field(default_factory=list)
    decisions: List[MemoryDecision] = Field(default_factory=list)
    persistence: List[PersistenceResult] = Field(default_factory=list)
    consolidation_memory_map: Dict[str, str] = Field(default_factory=dict)
    graph_relationships: List[GraphRelationshipResult] = Field(default_factory=list)


# Legacy one-stage contracts remain only so existing callers can migrate safely.
class MemoryActionType(str, Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    REJECT = "REJECT"


class RelationshipEntityType(str, Enum):
    MEMORY_ACTION = "memory_action"
    LONG_TERM_MEMORY = "long_term_memory"
    CANDIDATE_MEMORY = "candidate_memory"
    PATTERN = "pattern"
    GOAL = "goal"


class MemoryAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(..., min_length=1)
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
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1)
    source_type: RelationshipEntityType
    relation: RelationshipType
    target_id: str = Field(..., min_length=1)
    target_type: RelationshipEntityType
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class DataAgentResult(BaseModel):
    """Deprecated one-stage result retained for existing callers and tests."""

    model_config = ConfigDict(extra="forbid")

    user_id: Optional[str] = None
    memory_actions: List[MemoryAction] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    reasoning_summary: str = "No candidate memories were supplied."
