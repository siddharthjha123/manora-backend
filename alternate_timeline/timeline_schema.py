"""Pydantic contracts for task planning and alternate-timeline prediction."""

import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, model_validator


class TaskStatus(str, Enum):
    PLANNED = "planned"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskDecision(str, Enum):
    COMPLETE = "complete"
    SKIP = "skip"
    CANCEL = "cancel"


class CreateTaskRequest(BaseModel):
    """Task details entered by the student on Alternate Timeline."""

    user_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    date: datetime.date
    start_time: datetime.time
    end_time: datetime.time

    @model_validator(mode="after")
    def validate_time_order(self):
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time")
        return self


class TaskResponse(BaseModel):
    """Frontend representation of one planned or decided task."""

    task_id: str
    user_id: str
    title: str
    description: str
    date: datetime.date
    start_time: datetime.time
    end_time: datetime.time
    status: TaskStatus


class TaskListResponse(BaseModel):
    user_id: str
    date: datetime.date
    tasks: List[TaskResponse]


class TaskDecisionRequest(BaseModel):
    decision: TaskDecision
    reason: str = Field(default="", max_length=2000)


class TaskDecisionResponse(BaseModel):
    task_id: str
    decision: TaskDecision
    reason: str
    status: TaskStatus


class TimelinePredictionRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    scenario: TaskDecision


class TimelineScenario(BaseModel):
    task_id: str
    decision: TaskDecision
    reason: str


class TimelineBaseline(BaseModel):
    description: str = Field(..., min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class TimelineEvent(BaseModel):
    time: str = Field(..., min_length=1)
    event: str = Field(..., min_length=1)
    likely_effect: str = Field(..., min_length=1)


class TimelinePredictionContent(BaseModel):
    """Qwen output before task identity is attached by the service."""

    baseline: TimelineBaseline
    events: List[TimelineEvent] = Field(default_factory=list)
    summary: str = Field(..., min_length=1)


class TimelinePredictionResponse(BaseModel):
    """Read-only scenario prediction; the result is not persisted in V1."""

    scenario: TimelineScenario
    baseline: TimelineBaseline
    events: List[TimelineEvent]
    summary: str
