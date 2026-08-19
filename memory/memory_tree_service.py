"""Read-only Memory Tree application service."""

import json
import logging
from typing import Any, Dict, List, Optional

from data_agent.data_llm_client import DataAgentLLMClient, data_agent_llm_client
from database.connection import DatabaseManager, db
from memory.memory_tree_schema import (
    EmotionMemoriesResponse,
    MemoryTreeItem,
    MemoryTreeNode,
    MemoryTreeResponse,
    ReflectResponse,
    ReflectionContent,
)

logger = logging.getLogger("manora.memory.tree")


class MemoryTreeService:
    """Map internal emotions into five stable UI branches and reflect on them.

    This service only reads long-term memories. It intentionally does not call
    Data Agent Stage 1/2 or any persistence method, so opening or reflecting on
    the tree can never change the user's memory.
    """

    CATEGORIES = ("happy", "sad", "angry", "anxious", "calm")
    EMOTION_CATEGORY = {
        "joy": "happy",
        "happiness": "happy",
        "hope": "happy",
        "excitement": "happy",
        "motivation": "happy",
        "pride": "happy",
        "sadness": "sad",
        "loneliness": "sad",
        "isolation": "sad",
        "guilt": "sad",
        "hopelessness": "sad",
        "grief": "sad",
        "anger": "angry",
        "frustration": "angry",
        "irritation": "angry",
        "resentment": "angry",
        "anxiety": "anxious",
        "stress": "anxious",
        "worry": "anxious",
        "insecurity": "anxious",
        "fear": "anxious",
        "overwhelm": "anxious",
        "nervousness": "anxious",
        "calm": "calm",
        "relief": "calm",
        "contentment": "calm",
        "peace": "calm",
    }

    def __init__(
        self,
        database: DatabaseManager = db,
        reasoning_client: Optional[DataAgentLLMClient] = None,
    ):
        self.db = database
        self.reasoning_client = reasoning_client or data_agent_llm_client

    @classmethod
    def normalize_category(cls, emotion: str) -> str:
        """Validate and normalize a frontend branch name."""

        category = emotion.strip().lower()
        if category not in cls.CATEGORIES:
            raise ValueError(
                f"Unsupported Memory Tree emotion '{emotion}'. "
                f"Choose one of: {', '.join(cls.CATEGORIES)}"
            )
        return category

    @staticmethod
    def _decode_emotions(value: Any) -> List[Dict[str, Any]]:
        """Handle both asyncpg JSON values and legacy JSON strings."""

        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return []
        return [item for item in (value or []) if isinstance(item, dict)]

    @classmethod
    def _category_for_memory(cls, memory: Dict[str, Any]) -> Optional[str]:
        """Assign one memory to its strongest recognized emotion category."""

        emotions = cls._decode_emotions(memory.get("emotions"))
        ranked = sorted(
            emotions,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        )
        for item in ranked:
            category = cls.EMOTION_CATEGORY.get(
                str(item.get("emotion", "")).strip().lower()
            )
            if category:
                return category
        return None

    @staticmethod
    def _to_item(memory: Dict[str, Any]) -> MemoryTreeItem:
        return MemoryTreeItem(
            memory_id=str(memory["id"]),
            content=str(memory.get("content", "")),
            importance=float(memory.get("importance", 0.5)),
            confidence=float(memory.get("confidence", 0.5)),
        )

    async def _load_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Read active memories from PostgreSQL, the system of record."""

        return await self.db.get_long_term_memories(user_id=str(user_id), limit=500)

    async def get_tree(self, user_id: str) -> MemoryTreeResponse:
        """Return all five UI branches, including branches with zero memories."""

        memories = await self._load_memories(user_id)
        counts = {category: 0 for category in self.CATEGORIES}
        for memory in memories:
            category = self._category_for_memory(memory)
            if category:
                counts[category] += 1
        return MemoryTreeResponse(
            user_id=str(user_id),
            nodes=[
                MemoryTreeNode(emotion=category, memory_count=counts[category])
                for category in self.CATEGORIES
            ],
        )

    async def get_memories(
        self,
        user_id: str,
        emotion: str,
    ) -> EmotionMemoriesResponse:
        """Return memories assigned to one selected branch, ranked by importance."""

        category = self.normalize_category(emotion)
        memories = await self._load_memories(user_id)
        selected = [
            self._to_item(memory)
            for memory in memories
            if self._category_for_memory(memory) == category
        ]
        selected.sort(
            key=lambda item: (item.importance, item.confidence),
            reverse=True,
        )
        return EmotionMemoriesResponse(emotion=category, memories=selected)

    async def reflect(self, user_id: str, emotion: str) -> ReflectResponse:
        """Generate a read-only reflection over the selected branch's memories."""

        selected = await self.get_memories(user_id, emotion)
        memories = selected.memories[:8]
        if not memories:
            return ReflectResponse(
                emotion=selected.emotion,
                memories=[],
                reflection=ReflectionContent(
                    summary=(
                        f"There are no long-term memories in the "
                        f"{selected.emotion} branch yet."
                    ),
                    contributing_factors=[],
                ),
            )

        logger.info(
            "Generating read-only reflection for user %s branch %s using %d memories",
            user_id,
            selected.emotion,
            len(memories),
        )
        reflection = await self.reasoning_client.generate_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain patterns in existing student memories. Return "
                        "concise JSON matching the supplied schema. Do not diagnose, "
                        "invent facts, give medical advice, or propose memory writes."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "emotion_branch": selected.emotion,
                            "memories": [item.model_dump() for item in memories],
                        }
                    ),
                },
            ],
            schema=ReflectionContent,
            temperature=0.2,
            max_tokens=1200,
        )
        return ReflectResponse(
            emotion=selected.emotion,
            memories=memories,
            reflection=reflection,
        )


memory_tree_service = MemoryTreeService()
