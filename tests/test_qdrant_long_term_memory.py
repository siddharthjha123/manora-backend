"""Neon-to-Qdrant integration test for long-term memories."""

import json
import uuid
from typing import Any

import pytest

from database.connection import DatabaseManager
from vector_db.qdrant_client import QdrantAdapter


SEARCH_TEXT = "I don't have anyone I can really talk to."
MOCK_EVIDENCE_IDS = {"cm_019", "cm_020", "cm_021", "cm_022"}


def _decode_jsonb(value: Any) -> Any:
    """Decode JSONB values when asyncpg returns them as JSON strings."""

    if isinstance(value, str):
        return json.loads(value)
    return value


def _print_result(label: str, value: Any) -> None:
    print(f"\n{label}:")
    print(json.dumps(value, indent=2, default=str))


@pytest.mark.asyncio
async def test_neon_long_term_memory_qdrant_sync_and_user_isolation() -> None:
    """Sync an existing Neon memory, search it, and enforce user isolation."""

    database = DatabaseManager()
    qdrant = QdrantAdapter()
    memory_id = None
    previous_qdrant_point = None

    try:
        await database.initialize()

        assert database.is_postgres_connected, (
            "Neon PostgreSQL is not connected. Check DATABASE_URL and network access."
        )
        assert database._pool is not None
        assert qdrant.enabled, "QDRANT_ENABLED must be true for this integration test."
        assert qdrant.client is not None, (
            "Live Qdrant is not connected. Check QDRANT_URL, QDRANT_API_KEY, and network "
            "access. The in-memory fallback must not satisfy this integration test."
        )

        collection_names = {
            collection.name
            for collection in qdrant.client.get_collections().collections
        }
        assert qdrant.COLLECTION_NAME in collection_names
        print(f"\nQdrant connected: True")
        print(f"Collection exists: {qdrant.COLLECTION_NAME}")

        # Locate the existing long-term memory generated from the friendship/loneliness
        # mock interactions. Candidate numbering changed during development, so accept
        # both the requested cm_019/cm_020 and current cm_021/cm_022 evidence labels.
        async with database._pool.acquire() as connection:
            rows = await connection.fetch(
                """
                SELECT id, user_id, content, importance, confidence, emotions,
                       evidence_ids, created_at, updated_at, is_active
                FROM long_term_memories
                WHERE is_active = TRUE
                ORDER BY updated_at DESC
                """
            )

        matching_row = next(
            (
                dict(row)
                for row in rows
                if MOCK_EVIDENCE_IDS.intersection(
                    set(_decode_jsonb(row["evidence_ids"]))
                )
            ),
            None,
        )
        assert matching_row is not None, (
            "No active Neon long-term memory contains evidence from cm_019/cm_020 or "
            "cm_021/cm_022. Run the consolidation persistence step first."
        )

        memory_id = str(matching_row["id"])
        user_id = str(matching_row["user_id"])

        # Retrieve through the repository API after using SQL only to discover the owner.
        memory = await database.get_long_term_memory(
            memory_id=memory_id,
            user_id=user_id,
        )
        assert memory is not None
        _print_result("EXISTING NEON MEMORY", memory)

        # Preserve an already-synchronized point so the test can restore it exactly.
        existing_points = qdrant.client.retrieve(
            collection_name=qdrant.COLLECTION_NAME,
            ids=[memory_id],
            with_payload=True,
            with_vectors=True,
        )
        if existing_points:
            previous_qdrant_point = existing_points[0]

        metadata = {
            "importance": memory["importance"],
            "confidence": memory["confidence"],
            "emotions": _decode_jsonb(memory["emotions"]),
            "evidence_ids": _decode_jsonb(memory["evidence_ids"]),
            "is_active": memory["is_active"],
        }

        upserted = qdrant.upsert_memory(
            memory_id=memory_id,
            text=memory["content"],
            user_id=user_id,
            metadata=metadata,
        )
        assert upserted is True

        # Verify the point reached live Qdrant, rather than the adapter's fallback store.
        stored_points = qdrant.client.retrieve(
            collection_name=qdrant.COLLECTION_NAME,
            ids=[memory_id],
            with_payload=True,
        )
        assert len(stored_points) == 1
        stored_point = stored_points[0]
        stored_payload = stored_point.payload or {}
        _print_result(
            "QDRANT UPSERT",
            {"point_id": stored_point.id, "payload": stored_payload},
        )

        assert str(stored_point.id) == memory_id
        assert stored_payload["memory_id"] == memory_id
        assert stored_payload["user_id"] == user_id
        assert stored_payload["text"] == memory["content"]
        for key, expected_value in metadata.items():
            assert stored_payload[key] == expected_value

        results = qdrant.search_long_term_memories(
            user_id=user_id,
            query_text=SEARCH_TEXT,
            limit=5,
            score_threshold=0.0,
        )
        _print_result("SAME-USER SEARCH", results)

        matching_result = next(
            (result for result in results if result["memory_id"] == memory_id),
            None,
        )
        assert matching_result is not None
        assert matching_result["metadata"]["user_id"] == user_id
        for key, expected_value in metadata.items():
            assert matching_result["metadata"][key] == expected_value

        other_user_id = str(uuid.uuid4())
        isolated_results = qdrant.search_long_term_memories(
            user_id=other_user_id,
            query_text=SEARCH_TEXT,
            limit=5,
            score_threshold=0.0,
        )
        _print_result("OTHER-USER SEARCH", isolated_results)

        assert isolated_results == []

    finally:
        if memory_id is not None and qdrant.client is not None:
            if previous_qdrant_point is not None:
                from qdrant_client.http.models import PointStruct

                qdrant.client.upsert(
                    collection_name=qdrant.COLLECTION_NAME,
                    points=[
                        PointStruct(
                            id=previous_qdrant_point.id,
                            vector=previous_qdrant_point.vector,
                            payload=previous_qdrant_point.payload,
                        )
                    ],
                    wait=True,
                )
                print(f"\nCleanup: restored existing Qdrant point {memory_id}")
            else:
                from qdrant_client.http.models import PointIdsList

                qdrant.client.delete(
                    collection_name=qdrant.COLLECTION_NAME,
                    points_selector=PointIdsList(points=[memory_id]),
                    wait=True,
                )
                print(f"\nCleanup: deleted test Qdrant point {memory_id}")

        await database.close()
        print("Database connection closed; Neon memory was not modified.")
