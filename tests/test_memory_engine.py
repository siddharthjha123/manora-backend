"""
MANORA Tests - Memory Engine, Qdrant, and Neo4j.
Validates retrieval decisions, semantic search, graph relations, and candidate memory filtering.
"""

import pytest
from data_agent.data_schema import CandidateMemory, CandidateMemoryBehavior, CandidateMemoryDecision
from graph_db.neo4j_client import Neo4jAdapter
from memory.memory_engine import MemoryEngine
from vector_db.qdrant_client import QdrantAdapter


class TestMemoryRetrievalDecision:
    """Tests deterministic heuristic retrieval decision logic."""

    def test_should_not_retrieve_for_trivial_chat(self):
        engine = MemoryEngine()
        assert engine.should_retrieve("I'm sleepy.") is False
        assert engine.should_retrieve("hello there") is False
        assert engine.should_retrieve("good morning") is False

    def test_should_retrieve_for_recurrence(self):
        engine = MemoryEngine()
        assert engine.should_retrieve("This is happening again just like last month.") is True
        assert engine.should_retrieve("I keep repeating the same mistake.") is True

    def test_should_retrieve_for_goal_and_behavior(self):
        engine = MemoryEngine()
        assert engine.should_retrieve("I'm thinking about giving up on placements.") is True
        assert engine.should_retrieve("I planned to study but watched Netflix instead.") is True


class TestAdaptersAndPersistence:
    """Tests Qdrant vector adapter and Neo4j graph adapter operations."""

    def test_qdrant_adapter_crud(self):
        adapter = QdrantAdapter()
        user_id = "user-test-123"
        memory_id = "mem-test-456"

        # Upsert
        success = adapter.upsert_memory(
            memory_id=memory_id,
            text="Student struggles to follow planned study sessions when watching series.",
            user_id=user_id,
            metadata={"importance": 0.85},
        )
        assert success is True

        # Search
        results = adapter.search_memories(
            user_id=user_id,
            query_text="trouble studying due to entertainment",
            limit=3,
        )
        assert len(results) >= 1
        assert results[0]["memory_id"] == memory_id

        # Delete
        del_success = adapter.delete_memory(memory_id)
        assert del_success is True

    def test_neo4j_adapter_relationships(self):
        adapter = Neo4jAdapter()
        user_id = "user-graph-123"
        memory_id = "mem-graph-456"

        data = {
            "content": "Student avoided study schedule",
            "behavior": {"type": "avoidance", "description": "avoided study session"},
            "decision": {"description": "decided to play games instead"},
            "goal_relevance": {"related": True, "goal": "placement_preparation"},
            "emotional_state": [{"emotion": "guilt", "confidence": 0.8}],
        }

        created = adapter.create_memory_relationships(
            user_id=user_id,
            memory_id=memory_id,
            data=data,
        )
        assert created is True

        context = adapter.get_relevant_graph_context(user_id=user_id)
        assert len(context) >= 1
        assert context[0]["memory_id"] == memory_id

    @pytest.mark.asyncio
    async def test_persist_candidates_filtering(self):
        engine = MemoryEngine()
        user_id = "user-persist-test"
        interaction_id = "interaction-persist-test"

        candidates = [
            CandidateMemory(
                content="Student avoided planned study session due to distractions.",
                importance=0.85,
                confidence=0.90,
                behavior=CandidateMemoryBehavior(type="avoidance", description="avoided study"),
                decision=CandidateMemoryDecision(description="chose distraction"),
            ),
            CandidateMemory(
                content="Student mentioned weather is cloudy.",
                importance=0.20,
                confidence=0.40,
            ),
        ]

        persisted = await engine.persist_candidates(
            user_id=user_id,
            interaction_id=interaction_id,
            candidate_memories=candidates,
        )
        assert len(persisted) == 2
        # First one should be accepted because importance >= 0.60
        assert persisted[0]["status"] == "accepted"
        # Second one below threshold should remain pending
        assert persisted[1]["status"] == "pending"
