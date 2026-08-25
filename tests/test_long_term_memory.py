"""Integration test for the PostgreSQL long-term-memory repository methods."""

import json
import uuid
from typing import Any

import pytest

from database.connection import DatabaseManager


def _print_result(operation: str, value: Any) -> None:
    """Print an operation result in a form that also supports UUID/datetime values."""

    print(f"\n{operation}:")
    print(json.dumps(value, indent=2, default=str))


@pytest.mark.asyncio
async def test_long_term_memory_crud() -> None:
    """Create, read, list, and update one long-term memory in Neon."""

    database = DatabaseManager()
    test_user_id = str(uuid.uuid4())
    memory_id = None

    initial_content = "Integration test: student wants to build close friendships."
    updated_content = (
        "Integration test: student wants close friendships and someone they can trust."
    )

    try:
        await database.initialize()
        print(f"\nDatabase connected: {database.is_postgres_connected}")
        print(f"Test user UUID: {test_user_id}")

        assert database.is_postgres_connected, (
            "A PostgreSQL connection was not established. Check DATABASE_URL, network "
            "access, asyncpg installation, and Neon availability. The repository would "
            "otherwise use its in-memory fallback instead of testing Neon."
        )

        created = await database.create_long_term_memory(
            user_id=test_user_id,
            content=initial_content,
            importance=0.82,
            confidence=0.89,
            emotions=[{"emotion": "loneliness", "confidence": 0.91}],
            evidence_ids=["test-cm-019"],
        )
        _print_result("CREATE", created)

        assert created.get("id") is not None
        memory_id = str(created["id"])

        # The repository intentionally falls back to memory on SQL errors. Confirm the
        # created record really reached Neon so this cannot become a false-positive test.
        async with database._pool.acquire() as connection:
            persisted_in_postgres = await connection.fetchval(
                "SELECT EXISTS(SELECT 1 FROM long_term_memories WHERE id = $1::uuid)",
                uuid.UUID(memory_id),
            )
        assert persisted_in_postgres, (
            "create_long_term_memory() returned a record, but it was not persisted in "
            "PostgreSQL; inspect the logged SQL error and the Neon table schema."
        )

        retrieved = await database.get_long_term_memory(
            memory_id=memory_id,
            user_id=test_user_id,
        )
        _print_result("GET ONE", retrieved)

        assert retrieved is not None
        assert retrieved["content"] == initial_content

        memories = await database.get_long_term_memories(user_id=test_user_id)
        _print_result("GET MANY", memories)

        assert any(str(memory["id"]) == memory_id for memory in memories)

        updated = await database.update_long_term_memory(
            memory_id=memory_id,
            user_id=test_user_id,
            content=updated_content,
            importance=0.90,
            confidence=0.93,
            emotions=[{"emotion": "hope", "confidence": 0.84}],
            evidence_ids=["test-cm-019", "test-cm-020"],
        )
        _print_result("UPDATE", updated)

        assert updated is not None
        assert updated["content"] == updated_content

        retrieved_after_update = await database.get_long_term_memory(
            memory_id=memory_id,
            user_id=test_user_id,
        )
        _print_result("GET AFTER UPDATE", retrieved_after_update)

        assert retrieved_after_update is not None
        assert retrieved_after_update["content"] == updated_content

    finally:
        if memory_id is not None and database.is_postgres_connected and database._pool:
            async with database._pool.acquire() as connection:
                await connection.execute(
                    "DELETE FROM long_term_memories WHERE id = $1::uuid",
                    uuid.UUID(memory_id),
                )
            print(f"\nCleanup: removed test memory {memory_id}")

        await database.close()
        print("Database connection closed.")
