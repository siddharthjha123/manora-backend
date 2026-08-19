"""Thin HTTP routes for browsing and reflecting on the Memory Tree."""

from fastapi import APIRouter, HTTPException, status

from memory.memory_tree_schema import (
    EmotionMemoriesResponse,
    MemoryTreeResponse,
    ReflectRequest,
    ReflectResponse,
)
from memory.memory_tree_service import memory_tree_service

router = APIRouter(prefix="/memory-tree", tags=["Memory Tree"])


@router.get(
    "/{user_id}",
    response_model=MemoryTreeResponse,
    status_code=status.HTTP_200_OK,
)
async def get_memory_tree(user_id: str) -> MemoryTreeResponse:
    """Return counts for the Happy, Sad, Angry, Anxious, and Calm branches.

    Internal emotions such as loneliness, stress, frustration, and worry are
    deterministically mapped into the five frontend categories. The route never
    exposes PostgreSQL, Qdrant, or Neo4j implementation details.
    """

    try:
        return await memory_tree_service.get_tree(user_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load Memory Tree: {exc}",
        ) from exc


@router.get(
    "/{user_id}/emotions/{emotion}",
    response_model=EmotionMemoriesResponse,
    status_code=status.HTTP_200_OK,
)
async def get_memories_for_emotion(
    user_id: str,
    emotion: str,
) -> EmotionMemoriesResponse:
    """List the user's long-term memories for one Memory Tree branch."""

    try:
        return await memory_tree_service.get_memories(user_id, emotion)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load branch memories: {exc}",
        ) from exc


@router.post(
    "/{user_id}/reflect",
    response_model=ReflectResponse,
    status_code=status.HTTP_200_OK,
)
async def reflect_on_memory_branch(
    user_id: str,
    payload: ReflectRequest,
) -> ReflectResponse:
    """Generate a concise read-only reflection from existing branch memories.

    This endpoint deliberately never calls candidate extraction, Stage 1, Stage
    2, or persistence. Clicking Reflect cannot create or update a memory.
    """

    try:
        return await memory_tree_service.reflect(user_id, payload.emotion)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate reflection: {exc}",
        ) from exc
