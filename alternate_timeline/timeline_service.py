"""Application service for V1 Alternate Timeline task and prediction endpoints."""

import datetime
import json
import logging
from typing import Any, Dict, Optional

from alternate_timeline.timeline_schema import (
    CreateTaskRequest,
    TaskDecision,
    TaskDecisionRequest,
    TaskDecisionResponse,
    TaskListResponse,
    TaskResponse,
    TaskStatus,
    TimelinePredictionContent,
    TimelinePredictionRequest,
    TimelinePredictionResponse,
    TimelineScenario,
)
from data_agent.data_llm_client import DataAgentLLMClient, data_agent_llm_client
from database.connection import DatabaseManager, db
from memory.memory_engine import MemoryEngine, memory_engine

logger = logging.getLogger("manora.alternate_timeline")


class AlternateTimelineService:
    """Manage V1 tasks and generate read-only alternate timeline predictions.

    Task plans and decisions use the scheduled_tasks repository. Prediction
    context is read through MemoryEngine, and neither decisions nor predictions
    automatically become candidate or long-term memories.
    """

    DECISION_STATUS = {
        TaskDecision.COMPLETE: TaskStatus.COMPLETED,
        TaskDecision.SKIP: TaskStatus.SKIPPED,
        TaskDecision.CANCEL: TaskStatus.CANCELLED,
    }

    def __init__(
        self,
        database: DatabaseManager = db,
        memory: MemoryEngine = memory_engine,
        reasoning_client: Optional[DataAgentLLMClient] = None,
    ):
        self.database = database
        self.memory = memory
        self.reasoning_client = reasoning_client or data_agent_llm_client

    @staticmethod
    def _to_response(task: Dict[str, Any]) -> TaskResponse:
        return TaskResponse(
            task_id=str(task.get("task_id") or task["id"]),
            user_id=task["user_id"],
            title=task["title"],
            description=task["description"],
            date=task.get("date") or task["scheduled_date"],
            start_time=task["start_time"],
            end_time=task["end_time"],
            status=task["status"],
        )

    async def create_task(self, request: CreateTaskRequest) -> TaskResponse:
        """Create a planned task without creating a conversation or memory."""

        task = await self.database.create_scheduled_task(
            user_id=str(request.user_id),
            title=request.title,
            description=request.description,
            scheduled_date=request.date,
            start_time=request.start_time,
            end_time=request.end_time,
        )
        logger.info("Created Alternate Timeline task %s for user %s", task["id"], request.user_id)
        return self._to_response(task)

    async def list_tasks(
        self,
        user_id: str,
        selected_date: datetime.date,
    ) -> TaskListResponse:
        """Return one user's tasks for a date in chronological order."""

        records = await self.database.get_scheduled_tasks(str(user_id), selected_date)
        tasks = [self._to_response(task) for task in records]
        return TaskListResponse(user_id=str(user_id), date=selected_date, tasks=tasks)

    async def decide_task(
        self,
        task_id: str,
        request: TaskDecisionRequest,
    ) -> TaskDecisionResponse:
        """Record the explicit task decision without invoking memory processing."""

        status_value = self.DECISION_STATUS[request.decision]
        task = await self.database.update_scheduled_task_decision(
            task_id=str(task_id),
            decision=request.decision.value,
            reason=request.reason.strip(),
            status=status_value.value,
        )
        if task is None:
            raise KeyError(f"Task '{task_id}' was not found")
        return TaskDecisionResponse(
            task_id=str(task.get("task_id") or task["id"]),
            decision=request.decision,
            reason=task["reason"],
            status=task["status"],
        )

    async def predict(
        self,
        request: TimelinePredictionRequest,
    ) -> TimelinePredictionResponse:
        """Retrieve historical context and ask Qwen for a cautious scenario."""

        task = await self.database.get_scheduled_task(request.task_id)
        if task is None:
            raise KeyError(f"Task '{request.task_id}' was not found")
        if str(task["user_id"]) != str(request.user_id):
            raise PermissionError("The task does not belong to this user")
        task_snapshot = dict(task)

        query = " ".join(
            part
            for part in [
                task_snapshot["title"],
                task_snapshot["description"],
                request.scenario.value,
                task_snapshot.get("reason", ""),
            ]
            if part
        )
        context = await self.memory.retrieve_context(
            user_id=str(request.user_id),
            text=query,
        )
        logger.info(
            "Generating alternate timeline for task %s with %d semantic memories",
            request.task_id,
            len(context.get("memories") or []),
        )
        prediction = await self.reasoning_client.generate_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a cautious alternate timeline from the supplied "
                        "task and historical context. Return JSON matching the schema. "
                        "Describe likely possibilities, not certainties. Do not diagnose, "
                        "shame, or invent history."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": {
                                "title": task_snapshot["title"],
                                "description": task_snapshot["description"],
                                "date": str(
                                    task_snapshot.get("date")
                                    or task_snapshot["scheduled_date"]
                                ),
                                "start_time": str(task_snapshot["start_time"]),
                                "end_time": str(task_snapshot["end_time"]),
                            },
                            "decision": request.scenario.value,
                            "reason": task_snapshot.get("reason", ""),
                            "relevant_memories": context.get("memories") or [],
                            "graph_context": context.get("graph_context") or [],
                        },
                        default=str,
                    ),
                },
            ],
            schema=TimelinePredictionContent,
            temperature=0.2,
            max_tokens=1800,
        )
        return TimelinePredictionResponse(
            scenario=TimelineScenario(
                task_id=request.task_id,
                decision=request.scenario,
                reason=task_snapshot.get("reason", ""),
            ),
            baseline=prediction.baseline,
            events=prediction.events,
            summary=prediction.summary,
        )


alternate_timeline_service = AlternateTimelineService()
