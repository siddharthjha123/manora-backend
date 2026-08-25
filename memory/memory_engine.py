"""
MANORA Memory Engine.
Manages memory retrieval decisions, semantic/graph context aggregation,
and multi-database candidate memory persistence (PostgreSQL, Qdrant, Neo4j).
"""

import json
import logging
import re
import uuid
import time
from typing import Any, Dict, List, Optional

from data_agent.data_engine import DataAgent, DataAgentValidationError, data_agent
from data_agent.data_schema import (
    CandidateMemory,
    ExistingLongTermMemory,
    GraphRelationshipResult,
    GraphRelationshipStatus,
    MemoryDecision,
    MemoryDecisionType,
    MemoryPipelineResult,
    PersistenceResult,
    RetrievalResult,
)
from database.connection import DatabaseManager, db
from graph_db.neo4j_client import Neo4jAdapter, neo4j_adapter
from vector_db.qdrant_client import QdrantAdapter, qdrant_adapter
from observability.metrics import MEMORY_RETRIEVALS_TOTAL, MEMORY_RETRIEVAL_LATENCY_SECONDS

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
        database: Optional[DatabaseManager] = None,
        data_agent_instance: Optional[DataAgent] = None,
    ):
        self.qdrant = qdrant or qdrant_adapter
        self.neo4j = neo4j or neo4j_adapter
        self.db = database or db
        self.data_agent = data_agent_instance or data_agent

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

    def extract_candidate_memories(
        self,
        *,
        interaction: Dict[str, Any],
        emotion_analysis: Any,
    ) -> List[CandidateMemory]:
        """Extract and normalize temporary candidates for Stage 1 processing.

        The existing Data Agent extraction interface converts the Emotion Agent's
        structured analysis into ``CandidateMemory`` objects. Candidate IDs and
        ownership are completed here because Stage 1 requires stable evidence IDs
        and an explicit single-user boundary.
        """

        user_id = str(interaction.get("user_id") or "").strip()
        if not user_id:
            raise ValueError("Candidate-memory extraction requires interaction user_id")

        candidates = self.data_agent.process(interaction, emotion_analysis)
        for candidate in candidates:
            if candidate.user_id and str(candidate.user_id) != user_id:
                raise ValueError(
                    "Candidate-memory extraction returned a different user_id"
                )
            candidate.id = candidate.id or str(uuid.uuid4())
            candidate.user_id = user_id

        logger.info("Extracted %d candidate memories", len(candidates))
        return candidates

    async def retrieve_context(
        self,
        user_id: str,
        text: str,
    ) -> Dict[str, Any]:
        """
        Retrieves semantic memories from Qdrant and relationship context from Neo4j.
        """
        logger.info(f"Retrieving memory context for user {user_id}")
        start_time = time.perf_counter()

        try:
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

            elapsed = time.perf_counter() - start_time
            MEMORY_RETRIEVAL_LATENCY_SECONDS.labels(operation="retrieve_context").observe(elapsed)
            MEMORY_RETRIEVALS_TOTAL.labels(status="success", retrieval_performed="true").inc()

            return {
                "memories": semantic_memories,
                "graph_context": graph_context,
                "retrieval_performed": True,
            }
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            MEMORY_RETRIEVAL_LATENCY_SECONDS.labels(operation="retrieve_context").observe(elapsed)
            MEMORY_RETRIEVALS_TOTAL.labels(status="error", retrieval_performed="true").inc()
            logger.error("Failed to retrieve memory context: %s", exc)
            raise

    async def process_long_term_memories(
        self,
        candidate_memories: List[CandidateMemory],
        *,
        score_threshold: float = 0.0,
    ) -> MemoryPipelineResult:
        """Run both LLM stages, Qdrant retrieval, and final persistence."""

        if not candidate_memories:
            stage1 = await self.data_agent.consolidate_candidates([])
            return MemoryPipelineResult(stage1=stage1)

        user_id = candidate_memories[0].user_id or ""
        try:
            uuid.UUID(user_id)
        except (ValueError, TypeError) as exc:
            raise ValueError("Long-term-memory persistence requires a UUID user_id") from exc

        stage1 = await self.data_agent.consolidate_candidates(candidate_memories)
        self._validate_stage1_relationship_endpoints(stage1)
        retrievals: List[RetrievalResult] = []
        decisions: List[MemoryDecision] = []
        persistence: List[PersistenceResult] = []
        consolidation_memory_map: Dict[str, str] = {}

        for consolidated in stage1.consolidated_memories:
            logger.info("Qdrant: searching long-term memories")
            search_results = self.qdrant.search_long_term_memories(
                user_id=user_id,
                query_text=consolidated.content,
                limit=5,
                score_threshold=score_threshold,
            )
            existing_memories = self._qdrant_results_for_user(
                search_results,
                user_id=user_id,
            )
            logger.info("Qdrant: retrieved %d memories", len(existing_memories))

            retrievals.append(
                RetrievalResult(
                    consolidation_id=consolidated.consolidation_id,
                    query=consolidated.content,
                    existing_memories=[
                        memory.model_dump(mode="json")
                        for memory in existing_memories
                    ],
                )
            )
            decision = await self.data_agent.decide_memory_actions(
                user_id=user_id,
                consolidated_memory=consolidated,
                existing_long_term_memories=existing_memories,
            )
            decisions.append(decision)
            persistence_result = await self.persist_memory_decision(
                user_id=user_id,
                decision=decision,
            )
            persistence.append(persistence_result)
            if (
                persistence_result.memory_id
                and persistence_result.postgres_operation in {"CREATE", "UPDATE"}
                and persistence_result.qdrant_operation == "UPSERT"
            ):
                consolidation_memory_map[
                    consolidated.consolidation_id
                ] = persistence_result.memory_id

        graph_relationships = self._persist_stage1_relationships(
            user_id=user_id,
            stage1=stage1,
            decisions=decisions,
            consolidation_memory_map=consolidation_memory_map,
        )

        return MemoryPipelineResult(
            stage1=stage1,
            retrievals=retrievals,
            decisions=decisions,
            persistence=persistence,
            consolidation_memory_map=consolidation_memory_map,
            graph_relationships=graph_relationships,
        )

    @staticmethod
    def _validate_stage1_relationship_endpoints(stage1) -> None:
        """Refuse candidate IDs or any other non-consolidation relationship endpoint."""

        known_groups = {
            memory.consolidation_id for memory in stage1.consolidated_memories
        }
        for relationship in stage1.relationships:
            if (
                relationship.source_id not in known_groups
                or relationship.target_id not in known_groups
            ):
                raise DataAgentValidationError(
                    "MemoryEngine relationships must reference Stage 1 "
                    "consolidation_id values, never candidate memory IDs"
                )
            if relationship.source_id == relationship.target_id:
                raise DataAgentValidationError(
                    "MemoryEngine relationships cannot self-reference a consolidation"
                )

    def _persist_stage1_relationships(
        self,
        *,
        user_id: str,
        stage1,
        decisions: List[MemoryDecision],
        consolidation_memory_map: Dict[str, str],
    ) -> List[GraphRelationshipResult]:
        """Translate group relationships to final memory IDs and persist them last."""

        decision_by_group = {
            consolidated.consolidation_id: decision
            for consolidated, decision in zip(stage1.consolidated_memories, decisions)
        }

        # Neo4j contains only successfully persisted long-term memories.
        for consolidation_id, memory_id in consolidation_memory_map.items():
            decision = decision_by_group[consolidation_id]
            node_created = self.neo4j.upsert_memory_node(
                memory_id=memory_id,
                user_id=user_id,
                content=decision.content or "",
                importance=decision.importance,
                confidence=decision.confidence,
            )
            if node_created:
                self.neo4j.link_student_memory(
                    user_id=user_id,
                    memory_id=memory_id,
                )

        results: List[GraphRelationshipResult] = []
        for relationship in stage1.relationships:
            source_memory_id = consolidation_memory_map.get(relationship.source_id)
            target_memory_id = consolidation_memory_map.get(relationship.target_id)
            if not source_memory_id or not target_memory_id:
                results.append(
                    GraphRelationshipResult(
                        source_consolidation_id=relationship.source_id,
                        relation=relationship.relation,
                        target_consolidation_id=relationship.target_id,
                        source_memory_id=source_memory_id,
                        target_memory_id=target_memory_id,
                        evidence_ids=relationship.evidence_ids,
                        confidence=relationship.confidence,
                        status=GraphRelationshipStatus.SKIPPED_MISSING_ENDPOINT,
                    )
                )
                continue

            try:
                created = self.neo4j.create_memory_relationship(
                    user_id=user_id,
                    source_memory_id=source_memory_id,
                    relation=relationship.relation,
                    target_memory_id=target_memory_id,
                    evidence_ids=relationship.evidence_ids,
                    confidence=relationship.confidence,
                )
            except ValueError:
                logger.warning(
                    "Neo4j rejected translated relationship %s -> %s",
                    source_memory_id,
                    target_memory_id,
                    exc_info=True,
                )
                created = False

            results.append(
                GraphRelationshipResult(
                    source_consolidation_id=relationship.source_id,
                    relation=relationship.relation,
                    target_consolidation_id=relationship.target_id,
                    source_memory_id=source_memory_id,
                    target_memory_id=target_memory_id,
                    evidence_ids=relationship.evidence_ids,
                    confidence=relationship.confidence,
                    status=(
                        GraphRelationshipStatus.CREATED
                        if created
                        else GraphRelationshipStatus.REJECTED
                    ),
                )
            )
        return results

    def _qdrant_results_for_user(
        self,
        results: List[Dict[str, Any]],
        *,
        user_id: str,
    ) -> List[ExistingLongTermMemory]:
        """Convert same-user Qdrant hits into the Stage 2 input contract."""

        memories: List[ExistingLongTermMemory] = []
        for result in results:
            metadata = dict(result.get("metadata") or {})
            result_user_id = str(metadata.get("user_id", ""))
            if result_user_id != user_id:
                logger.warning("Discarded a cross-user Qdrant search result")
                continue

            memory_id = result.get("memory_id") or metadata.get("memory_id")
            if not memory_id:
                continue
            memories.append(
                ExistingLongTermMemory(
                    id=str(memory_id),
                    user_id=result_user_id,
                    content=result.get("text") or metadata.get("text", ""),
                    evidence_ids=self._decode_json_list(
                        metadata.get("evidence_ids", [])
                    ),
                    emotions=self._decode_json_list(metadata.get("emotions", [])),
                    importance=metadata.get("importance", 0.5),
                    confidence=metadata.get("confidence", 0.5),
                    metadata={**metadata, "score": result.get("score")},
                )
            )
        return memories

    @staticmethod
    def _decode_json_list(value: Any) -> List[Any]:
        if isinstance(value, str):
            value = json.loads(value)
        return list(value or [])

    async def persist_memory_decision(
        self,
        *,
        user_id: str,
        decision: MemoryDecision,
    ) -> PersistenceResult:
        """Execute one Stage 2 decision in PostgreSQL, then synchronize Qdrant."""

        if decision.action == MemoryDecisionType.REJECT:
            return PersistenceResult(
                action=decision.action,
                postgres_operation="NONE",
                qdrant_operation="NONE",
            )

        emotions = [emotion.model_dump() for emotion in decision.emotions]
        evidence_ids = list(decision.evidence_ids)
        if decision.action == MemoryDecisionType.CREATE:
            logger.info("Persistence: creating PostgreSQL memory")
            memory = await self.db.create_long_term_memory(
                user_id=user_id,
                content=decision.content or "",
                importance=decision.importance,
                confidence=decision.confidence,
                emotions=emotions,
                evidence_ids=evidence_ids,
            )
            postgres_operation = "CREATE"
        else:
            logger.info(
                "Persistence: updating PostgreSQL memory %s",
                decision.memory_id,
            )
            existing_memory = await self.db.get_long_term_memory(
                memory_id=decision.memory_id,
                user_id=user_id,
            )
            if existing_memory is None:
                return PersistenceResult(
                    action=decision.action,
                    memory_id=decision.memory_id,
                    postgres_operation="NOT_FOUND",
                    qdrant_operation="NONE",
                )

            existing_evidence = self._decode_json_list(
                existing_memory.get("evidence_ids", [])
            )
            evidence_ids = list(
                dict.fromkeys(existing_evidence + decision.evidence_ids)
            )
            memory = await self.db.update_long_term_memory(
                memory_id=decision.memory_id,
                user_id=user_id,
                content=decision.content or "",
                importance=decision.importance,
                confidence=decision.confidence,
                emotions=emotions,
                evidence_ids=evidence_ids,
            )
            postgres_operation = "UPDATE"

        if memory is None:
            return PersistenceResult(
                action=decision.action,
                memory_id=decision.memory_id,
                postgres_operation="NOT_FOUND",
                qdrant_operation="NONE",
            )

        memory_id = str(memory["id"])
        logger.info("Persistence: upserting Qdrant memory %s", memory_id)
        qdrant_upserted = self.qdrant.upsert_memory(
            memory_id=memory_id,
            text=decision.content or "",
            user_id=user_id,
            metadata={
                "importance": decision.importance,
                "confidence": decision.confidence,
                "emotions": emotions,
                "evidence_ids": evidence_ids,
                "is_active": bool(memory.get("is_active", True)),
            },
        )
        return PersistenceResult(
            action=decision.action,
            memory_id=memory_id,
            postgres_operation=postgres_operation,
            qdrant_operation="UPSERT" if qdrant_upserted else "FAILED",
            memory=memory,
        )


# Global singleton instance
memory_engine = MemoryEngine()
