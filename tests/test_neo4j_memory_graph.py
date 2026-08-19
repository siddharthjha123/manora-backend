"""Deterministic tests for the Neo4j V1 long-term-memory graph operations."""

import pytest

from graph_db.neo4j_client import Neo4jAdapter


USER_ID = "550e8400-e29b-41d4-a716-446655440000"
OTHER_USER_ID = "650e8400-e29b-41d4-a716-446655440000"
SOURCE_MEMORY_ID = "11111111-1111-4111-8111-111111111111"
TARGET_MEMORY_ID = "22222222-2222-4222-8222-222222222222"
OTHER_SOURCE_MEMORY_ID = "33333333-3333-4333-8333-333333333333"
OTHER_TARGET_MEMORY_ID = "44444444-4444-4444-8444-444444444444"


def offline_adapter() -> Neo4jAdapter:
    """Use the real adapter logic with its deterministic in-memory fallback."""

    adapter = Neo4jAdapter()
    adapter.close()
    return adapter


def seed_memory(adapter, memory_id, user_id):
    return adapter.upsert_memory_node(
        memory_id=memory_id,
        user_id=user_id,
        content=f"Memory {memory_id}",
        importance=0.8,
        confidence=0.9,
    )


def test_upsert_memory_node_updates_without_duplicate_nodes():
    adapter = offline_adapter()

    assert adapter.upsert_memory_node(
        memory_id=SOURCE_MEMORY_ID,
        user_id=USER_ID,
        content="Student sometimes feels socially isolated.",
        importance=0.75,
        confidence=0.80,
    )
    assert adapter.upsert_memory_node(
        memory_id=SOURCE_MEMORY_ID,
        user_id=USER_ID,
        content="Student persistently feels socially isolated.",
        importance=0.86,
        confidence=0.91,
    )

    assert len(adapter._in_memory_memories) == 1
    stored = adapter._in_memory_memories[SOURCE_MEMORY_ID]
    assert stored["content"] == "Student persistently feels socially isolated."
    assert stored["importance"] == 0.86
    assert stored["confidence"] == 0.91


def test_link_student_memory_is_idempotent():
    adapter = offline_adapter()
    seed_memory(adapter, SOURCE_MEMORY_ID, USER_ID)

    assert adapter.link_student_memory(USER_ID, SOURCE_MEMORY_ID)
    assert adapter.link_student_memory(USER_ID, SOURCE_MEMORY_ID)

    assert adapter._in_memory_student_memories[USER_ID] == {SOURCE_MEMORY_ID}


def test_memory_node_ownership_cannot_be_transferred_between_users():
    adapter = offline_adapter()
    seed_memory(adapter, SOURCE_MEMORY_ID, USER_ID)

    assert adapter.upsert_memory_node(
        memory_id=SOURCE_MEMORY_ID,
        user_id=OTHER_USER_ID,
        content="Attempted cross-user overwrite",
        importance=0.9,
        confidence=0.9,
    ) is False
    assert adapter.link_student_memory(OTHER_USER_ID, SOURCE_MEMORY_ID) is False
    assert adapter._in_memory_memories[SOURCE_MEMORY_ID]["user_id"] == USER_ID


def test_create_and_get_memory_relationship_is_idempotent():
    adapter = offline_adapter()
    seed_memory(adapter, SOURCE_MEMORY_ID, USER_ID)
    seed_memory(adapter, TARGET_MEMORY_ID, USER_ID)

    assert adapter.create_memory_relationship(
        user_id=USER_ID,
        source_memory_id=SOURCE_MEMORY_ID,
        relation="TRIGGERS",
        target_memory_id=TARGET_MEMORY_ID,
        evidence_ids=["cm_021"],
        confidence=0.85,
    )
    assert adapter.create_memory_relationship(
        user_id=USER_ID,
        source_memory_id=SOURCE_MEMORY_ID,
        relation="TRIGGERS",
        target_memory_id=TARGET_MEMORY_ID,
        evidence_ids=["cm_021", "cm_023", "cm_023"],
        confidence=0.90,
    )

    relationships = adapter.get_memory_relationships(USER_ID, SOURCE_MEMORY_ID)

    assert relationships == [
        {
            "source_memory_id": SOURCE_MEMORY_ID,
            "relation": "TRIGGERS",
            "target_memory_id": TARGET_MEMORY_ID,
            "confidence": 0.90,
            "evidence_ids": ["cm_021", "cm_023"],
        }
    ]
    assert adapter.get_memory_relationships(USER_ID, TARGET_MEMORY_ID) == relationships


def test_create_memory_relationship_rejects_unknown_relation():
    adapter = offline_adapter()

    with pytest.raises(ValueError, match="Unsupported memory relationship"):
        adapter.create_memory_relationship(
            user_id=USER_ID,
            source_memory_id=SOURCE_MEMORY_ID,
            relation="DELETES_EVERYTHING",
            target_memory_id=TARGET_MEMORY_ID,
            evidence_ids=["cm_021"],
            confidence=0.85,
        )


def test_create_memory_relationship_rejects_self_reference():
    adapter = offline_adapter()

    with pytest.raises(ValueError, match="cannot point to itself"):
        adapter.create_memory_relationship(
            user_id=USER_ID,
            source_memory_id=SOURCE_MEMORY_ID,
            relation="RELATED_TO",
            target_memory_id=SOURCE_MEMORY_ID,
            evidence_ids=["cm_021"],
            confidence=0.85,
        )


def test_create_memory_relationship_rejects_cross_user_endpoints():
    adapter = offline_adapter()
    seed_memory(adapter, SOURCE_MEMORY_ID, USER_ID)
    seed_memory(adapter, OTHER_TARGET_MEMORY_ID, OTHER_USER_ID)

    created = adapter.create_memory_relationship(
        user_id=USER_ID,
        source_memory_id=SOURCE_MEMORY_ID,
        relation="TRIGGERS",
        target_memory_id=OTHER_TARGET_MEMORY_ID,
        evidence_ids=["cm_021"],
        confidence=0.85,
    )

    assert created is False
    assert adapter.get_memory_relationships(USER_ID) == []
    assert adapter.get_memory_relationships(OTHER_USER_ID) == []


def test_get_memory_relationships_never_returns_another_users_graph():
    adapter = offline_adapter()
    seed_memory(adapter, SOURCE_MEMORY_ID, USER_ID)
    seed_memory(adapter, TARGET_MEMORY_ID, USER_ID)
    seed_memory(adapter, OTHER_SOURCE_MEMORY_ID, OTHER_USER_ID)
    seed_memory(adapter, OTHER_TARGET_MEMORY_ID, OTHER_USER_ID)

    assert adapter.create_memory_relationship(
        user_id=USER_ID,
        source_memory_id=SOURCE_MEMORY_ID,
        relation="SUPPORTS",
        target_memory_id=TARGET_MEMORY_ID,
        evidence_ids=["cm_a"],
        confidence=0.80,
    )
    assert adapter.create_memory_relationship(
        user_id=OTHER_USER_ID,
        source_memory_id=OTHER_SOURCE_MEMORY_ID,
        relation="CONTRADICTS",
        target_memory_id=OTHER_TARGET_MEMORY_ID,
        evidence_ids=["cm_b"],
        confidence=0.75,
    )

    user_relationships = adapter.get_memory_relationships(USER_ID)
    other_user_relationships = adapter.get_memory_relationships(OTHER_USER_ID)

    assert len(user_relationships) == 1
    assert user_relationships[0]["source_memory_id"] == SOURCE_MEMORY_ID
    assert user_relationships[0]["target_memory_id"] == TARGET_MEMORY_ID
    assert len(other_user_relationships) == 1
    assert other_user_relationships[0]["source_memory_id"] == OTHER_SOURCE_MEMORY_ID
    assert other_user_relationships[0]["target_memory_id"] == OTHER_TARGET_MEMORY_ID
