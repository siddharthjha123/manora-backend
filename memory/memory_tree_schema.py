"""Validated request and response contracts for the Memory Tree feature."""

from typing import List

from pydantic import BaseModel, Field


class MemoryTreeNode(BaseModel):
    """One stable frontend branch and the number of memories assigned to it."""

    emotion: str
    memory_count: int = Field(ge=0)


class MemoryTreeResponse(BaseModel):
    """Five-branch overview returned when the Memory Tree screen opens."""

    user_id: str
    nodes: List[MemoryTreeNode]


class MemoryTreeItem(BaseModel):
    """Small, frontend-safe representation of one long-term memory."""

    memory_id: str
    content: str
    importance: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class EmotionMemoriesResponse(BaseModel):
    """Memories assigned to one of the five UI emotion categories."""

    emotion: str
    memories: List[MemoryTreeItem]


class ReflectRequest(BaseModel):
    """Emotion branch selected by the user for read-only reflection."""

    emotion: str = Field(..., min_length=1)


class ReflectionContent(BaseModel):
    """Concise Qwen explanation derived only from existing memories."""

    summary: str = Field(..., min_length=1)
    contributing_factors: List[str] = Field(default_factory=list)


class ReflectResponse(BaseModel):
    """Read-only reflection response; it never creates or updates memory."""

    emotion: str
    memories: List[MemoryTreeItem]
    reflection: ReflectionContent
