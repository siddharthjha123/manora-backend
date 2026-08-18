"""
MANORA Memory Engine.
Manages memory retrieval decisions, semantic/graph context aggregation,
and multi-database candidate memory persistence (PostgreSQL, Qdrant, Neo4j).
"""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from data_agent.data_schema import CandidateMemory, DataAgentResult, MemoryActionType
from database.connection import db
from graph_db.neo4j_client import Neo4jAdapter, neo4j_adapter
from vector_db.qdrant_client import QdrantAdapter, qdrant_adapter

logger = logging.getLogger("manora.memory.engine")


class MemoryEngine:
    """Coordinates memory retrieval and candidate memory persistence across storage layers."""

    # Heuristic triggers for memory retrieval
    RECURRENCE_KEYWORDS = [
        "again", "repeating", "keeps happening", "every time", "last month", "last week",
        "always do this", "same mistake", "pattern", "as usual", "before", "history",
    ]

    GOAL_KEYWORDS = [
        "placement", "study", "exam", "grade", "assignment", "interview", "fail",
        "pass", "career", "give up", "give up on", "goal", "deadline", "internship",
        "cgpa", "project", "degree", "future", "coursework",
    ]

    BEHAVIORAL_KEYWORDS = [
        "planned to", "instead", "procrastinating", "procrastinated", "netflix", "watching",
        "delayed", "skipped", "avoided", "wasted time", "decided to", "scrolling",
        "series", "youtube", "gaming", "distracted",
    ]

    IMPORTANCE_THRESHOLD = 0.60
    CONFIDENCE_THRESHOLD = 0.60

    def __init__(
        self,
        qdrant: Optional[QdrantAdapter] = None,
        neo4j: Optional[Neo4jAdapter] = None,
    ):
        self.qdrant = qdrant or qdrant_adapter
        self.neo4j = neo4j or neo4j_adapter

    def should_retrieve(
        self,
        text: str,
        user_goals: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Deterministically evaluates whether historical context is likely useful.
        Avoids running expensive graph/vector searches for trivial chit-chat or simple status messages.
        """
        if not text or not text.strip():
            return False

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        # Very short message without significant keywords
        if len(words) < 4:
            if not any(kw in text_lower for kw in ["again", "fail", "placements", "goal"]):
                return False

        # 1. Check recurrence phrases
        for kw in self.RECURRENCE_KEYWORDS:
            if kw in text_lower:
                logger.debug(f"Memory retrieval triggered by recurrence phrase: '{kw}'")
                return True

        # 2. Check goal-related terms
        for kw in self.GOAL_KEYWORDS:
            if kw in text_lower:
                logger.debug(f"Memory retrieval triggered by goal term: '{kw}'")
                return True

        # 3. Check behavioral/decision terms
        for kw in self.BEHAVIORAL_KEYWORDS:
            if kw in text_lower:
                logger.debug(f"Memory retrieval triggered by behavioral marker: '{kw}'")
                return True

        # 4. Check active goal titles if provided
        if user_goals:
            for g in user_goals:
                title = g.get("title", "").lower()
                if title and title in text_lower:
                    logger.debug(f"Memory retrieval triggered by active goal match: '{title}'")
                    return True

        return False

    def empty_context(self) -> Dict[str, Any]:
        """Returns standard empty context dictionary."""
        return {
            "memories": [],
            "graph_context": [],
            "retrieval_performed": False,
        }

    async def retrieve_context(
        self,
        user_id: str,
        text: str,
    ) -> Dict[str, Any]:
        """
        Retrieves semantic memories from Qdrant and relationship context from Neo4j.
        """
        logger.info(f"Retrieving memory context for user {user_id}")

        # 1. Semantic search in Qdrant
        semantic_memories = self.qdrant.search_memories(
            user_id=user_id,
            query_text=text,
            limit=3,
        )

        # 2. Relationship search in Neo4j
        graph_context = self.neo4j.get_relevant_graph_context(
            user_id=user_id,
            limit=3,
        )

        return {
            "memories": semantic_memories,
            "graph_context": graph_context,
            "retrieval_performed": True,
        }

    async def persist_candidates(
        self,
        user_id: str,
        interaction_id: str,
        candidate_memories: List[CandidateMemory],
    ) -> List[Dict[str, Any]]:
        """
        Filters candidate memories by importance & confidence thresholds,
        and persists accepted memories to PostgreSQL, Qdrant, and Neo4j.
        """
        if not candidate_memories:
            return []

        persisted = []
        for candidate in candidate_memories:
            # Check meaningfulness threshold
            is_meaningful = (
                candidate.importance >= self.IMPORTANCE_THRESHOLD
                and candidate.confidence >= self.CONFIDENCE_THRESHOLD
            )

            status = "accepted" if is_meaningful else "pending"
            memory_id = candidate.id or str(uuid.uuid4())
            candidate.id = memory_id

            mem_dict = {
                "id": memory_id,
                "content": candidate.content,
                "context": candidate.context,
                "emotional_state": [e.model_dump() for e in candidate.emotional_state],
                "importance": candidate.importance,
                "confidence": candidate.confidence,
                "status": status,
            }

            # 1. Canonical write to PostgreSQL
            await db.save_candidate_memories(
                interaction_id=interaction_id,
                user_id=user_id,
                candidate_memories=[mem_dict],
            )

            # 2. Write to Vector & Graph layers if meaningful
            if is_meaningful:
                # Upsert into Qdrant for future semantic similarity search
                self.qdrant.upsert_memory(
                    memory_id=memory_id,
                    text=candidate.content,
                    user_id=user_id,
                    metadata={
                        "importance": candidate.importance,
                        "context": candidate.context,
                    },
                )

                # Link in Neo4j graph
                self.neo4j.create_memory_relationships(
                    user_id=user_id,
                    memory_id=memory_id,
                    data={
                        "content": candidate.content,
                        "emotional_state": [e.model_dump() for e in candidate.emotional_state],
                        "behavior": candidate.behavior.model_dump() if candidate.behavior else {},
                        "decision": candidate.decision.model_dump() if candidate.decision else {},
                        "goal_relevance": candidate.goal_relevance.model_dump() if candidate.goal_relevance else {},
                    },
                )

            persisted.append(mem_dict)

        logger.info(f"Processed {len(persisted)} candidate memories for interaction {interaction_id}")
        return persisted

    # --------------------------------------------------------------------
    # Beyond the above code is the actual data agent code 
    # --------------------------------------------------------------------
    async def persist_memory_candidates(self, result:DataAgentResult) -> List[Dict[str, Any]]:
        """
        Persist the long-term memory actions produced by the Data Agent.

        The Data Agent decides what should happen:
            CREATE  -> create a new long-term memory
            UPDATE  -> update an existing long-term memory
            MERGE   -> create one new consolidated memory
            REJECT  -> do nothing

        This method only executes those decisions in PostgreSQL.
        It strictly does not perform any kind of reasoning.
        """

        if not result.memory_actions:
            logger.info("No memory actions to persist")
            return []

        if not result.user_id:
            raise ValueError("DataAgentResult.user_id is required for persistence")

        persisted_memories: List[Dict[str, Any]] = []

        for action in result.memory_actions:
            try:
                # -------------------------------------------------
                # REJECT
                # -------------------------------------------------
                if action.action == MemoryActionType.REJECT:
                    logger.info(
                        "Rejected memory action %s: %s",
                        action.action_id,
                        action.reasoning,
                    )
                    continue

                # CREATE and MERGE both create a new consolidated memory.
                if action.action in {
                    MemoryActionType.CREATE,
                    MemoryActionType.MERGE,
                }:
                    memory = await db.create_long_term_memory(
                        user_id=result.user_id,
                        content=action.content,
                        importance=action.importance,
                        confidence=action.confidence,
                        emotions=[
                            emotion.model_dump()
                            for emotion in action.emotions
                        ],
                        evidence_ids=action.evidence_ids,
                    )

                    persisted_memories.append(
                        {
                            "action_id": action.action_id,
                            "action": action.action.value,
                            "memory": memory,
                        }
                    )

                    logger.info(
                        "%s action %s persisted as new long-term memory %s",
                        action.action.value,
                        action.action_id,
                        memory.get("id") if memory else None,
                    )

                    continue

                # -------------------------------------------------
                # UPDATE
                # -------------------------------------------------
                if action.action == MemoryActionType.UPDATE:
                    memory = await db.update_long_term_memory(
                        memory_id=action.memory_id,
                        user_id=result.user_id,
                        content=action.content,
                        importance=action.importance,
                        confidence=action.confidence,
                        emotions=[
                            emotion.model_dump()
                            for emotion in action.emotions
                        ],
                        evidence_ids=action.evidence_ids,
                    )

                    if memory is None:
                        logger.warning(
                            "UPDATE action %s could not find memory %s",
                            action.action_id,
                            action.memory_id,
                        )

                        persisted_memories.append(
                            {
                                "action_id": action.action_id,
                                "action": action.action.value,
                                "memory": None,
                                "status": "memory_not_found",
                            }
                        )

                        continue

                    persisted_memories.append(
                        {
                            "action_id": action.action_id,
                            "action": action.action.value,
                            "memory": memory,
                        }
                    )

                    logger.info(
                        "UPDATE action %s persisted for memory %s",
                        action.action_id,
                        action.memory_id,
                    )
            except Exception:
                logger.exception(
                    "Failed to persist memory action %s",
                    action.action_id,
                )

                persisted_memories.append(
                    {
                        "action_id": action.action_id,
                        "action": action.action.value,
                        "memory": None,
                        "status": "persistence_failed",
                    }
                )

        logger.info(
            "Persisted %d/%d Data Agent memory actions.",
            len(persisted_memories),
            len(result.memory_actions),
        )

        return persisted_memories


# Global singleton instance
memory_engine = MemoryEngine()
