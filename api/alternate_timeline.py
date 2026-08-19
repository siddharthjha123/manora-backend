"""Thin HTTP routes for Alternate Timeline task planning and prediction."""

import datetime

from fastapi import APIRouter, HTTPException, Query, status

from alternate_timeline.timeline_schema import (
    CreateTaskRequest,
    TaskDecisionRequest,
    TaskDecisionResponse,
    TaskListResponse,
    TaskResponse,
    TimelinePredictionRequest,
    TimelinePredictionResponse,
)
from alternate_timeline.timeline_service import alternate_timeline_service

router = APIRouter(prefix="/alternate-timeline", tags=["Alternate Timeline"])


@router.post(
    "/tasks",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_timeline_task(payload: CreateTaskRequest) -> TaskResponse:
    """Create a planned task displayed on the Alternate Timeline screen.

    This is task management, not conversation. The task is not inserted into the
    interactions table and does not automatically become a memory.
    """

    return await alternate_timeline_service.create_task(payload)


@router.get(
    "/tasks/{user_id}",
    response_model=TaskListResponse,
    status_code=status.HTTP_200_OK,
)
async def get_timeline_tasks(
    user_id: str,
    date: datetime.date = Query(..., description="Timeline date in YYYY-MM-DD format"),
) -> TaskListResponse:
    """Return the user's tasks for the requested calendar date."""

    return await alternate_timeline_service.list_tasks(user_id, date)


@router.post(
    "/tasks/{task_id}/decision",
    response_model=TaskDecisionResponse,
    status_code=status.HTTP_200_OK,
)
async def decide_timeline_task(
    task_id: str,
    payload: TaskDecisionRequest,
) -> TaskDecisionResponse:
    """Record complete, skip, or cancel for one task.

    A short reason such as "too tired" remains task-local. It is not sent into
    the Data Agent memory pipeline, preventing trivial scheduling choices from
    polluting long-term memory.
    """

    try:
        return await alternate_timeline_service.decide_task(task_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/predict",
    response_model=TimelinePredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def predict_alternate_timeline(
    payload: TimelinePredictionRequest,
) -> TimelinePredictionResponse:
    """Generate a read-only likely-outcomes timeline from historical context.

    MemoryEngine retrieves same-user semantic and graph context. Qwen reasons
    over that context, but this endpoint never creates or updates memories and
    does not persist the generated prediction in V1.
    """

    try:
        return await alternate_timeline_service.predict(payload)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate alternate timeline: {exc}",
        ) from exc
