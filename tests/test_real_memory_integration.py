"""Live integration test for PostgreSQL, Qdrant, and Neo4j memory identity."""

import uuid

import pytest

from data_agent.data_schema import (
    CandidateMemory,
    ConsolidatedMemory,
    ConsolidationRelationship,
    GraphRelationshipStatus,
    MemoryDecision,
    MemoryDecisionType,
    Stage1ConsolidationResult,
)
from database.connection import DatabaseManager
from graph_db.neo4j_client import Neo4jAdapter
from memory.memory_engine import MemoryEngine
from vector_db.qdrant_client import QdrantAdapter


class TwoMemoryCreateAgent:
    """Deterministic Stage 1/2 reasoning fixture; all storage remains live."""

    async def consolidate_candidates(self, candidate_memories):
        user_id = candidate_memories[0].user_id
        return Stage1ConsolidationResult(
            user_id=user_id,
            consolidated_memories=[
                ConsolidatedMemory(
                    consolidation_id="integration-group-a",
                    candidate_ids=["integration-rel-cm-a"],
                    evidence_ids=["integration-rel-cm-a"],
                    content="Student feels isolated when studying alone.",
                    emotions=[{"emotion": "loneliness", "confidence": 0.88}],
                    importance=0.82,
                    confidence=0.90,
                ),
                ConsolidatedMemory(
                    consolidation_id="integration-group-b",
                    candidate_ids=["integration-rel-cm-b"],
                    evidence_ids=["integration-rel-cm-b"],
                    content="Student seeks trusted support after prolonged isolation.",
                    emotions=[{"emotion": "hope", "confidence": 0.84}],
                    importance=0.86,
                    confidence=0.91,
                ),
            ],
            relationships=[
                ConsolidationRelationship(
                    source_id="integration-group-a",
                    relation="TRIGGERS",
                    target_id="integration-group-b",
                    evidence_ids=[
                        "integration-rel-cm-a",
                        "integration-rel-cm-b",
                    ],
                    confidence=0.87,
                )
            ],
            reasoning_summary="Two groups with one evidence-backed relationship.",
        )

    async def decide_memory_actions(
        self,
        *,
        user_id,
        consolidated_memory,
        existing_long_term_memories,
    ):
        del user_id, existing_long_term_memories
        return MemoryDecision(
            action=MemoryDecisionType.CREATE,
            candidate_ids=consolidated_memory.candidate_ids,
            evidence_ids=consolidated_memory.evidence_ids,
            content=consolidated_memory.content,
            emotions=consolidated_memory.emotions,
            importance=consolidated_memory.importance,
            confidence=consolidated_memory.confidence,
            reasoning="No existing memory is used in this CREATE integration fixture.",
        )


@pytest.mark.asyncio
async def test_real_long_term_memory_persistence() -> None:
    """Persist and verify the same long-term-memory UUID in all three stores."""

    user_id = str(uuid.uuid4())
    memory_id = None
    content = (
        "Integration test memory: student wants to build "
        "strong machine learning skills."
    )
    importance = 0.85
    confidence = 0.92
    emotions = [{"emotion": "motivation", "confidence": 0.90}]
    evidence_ids = ["integration-test-cm-001"]

    # Isolated instances prevent cleanup from closing the application's global DB.
    database = DatabaseManager()
    qdrant = QdrantAdapter()
    neo4j = Neo4jAdapter()

    try:
        await database.initialize()
        assert database.is_postgres_connected and database._pool is not None, (
            "Neon PostgreSQL is not connected. Check DATABASE_URL and network access."
        )
        assert qdrant.client is not None, (
            "Qdrant is not configured. Check QDRANT_ENABLED, QDRANT_URL, and "
            "QDRANT_API_KEY."
        )
        assert neo4j._driver is not None, (
            "Neo4j is not configured. Check NEO4J_ENABLED and its credentials."
        )

        # Construction can be lazy, so explicitly verify both remote clients.
        qdrant.client.get_collection(qdrant.COLLECTION_NAME)
        neo4j._driver.verify_connectivity()

        print("\n========== REAL INTEGRATION TEST ==========")
        print("User ID:", user_id)

        print("\n[1] PostgreSQL CREATE")
        postgres_memory = await database.create_long_term_memory(
            user_id=user_id,
            content=content,
            importance=importance,
            confidence=confidence,
            emotions=emotions,
            evidence_ids=evidence_ids,
        )
        assert postgres_memory is not None
        assert postgres_memory.get("id") is not None
        memory_id = str(postgres_memory["id"])

        # Repository methods can fall back after SQL errors. This direct query proves
        # that the record really reached PostgreSQL.
        async with database._pool.acquire() as connection:
            postgres_row = await connection.fetchrow(
                """
                SELECT id, user_id, content
                FROM long_term_memories
                WHERE id = $1::uuid AND user_id = $2::uuid
                """,
                uuid.UUID(memory_id),
                uuid.UUID(user_id),
            )
        assert postgres_row is not None
        assert str(postgres_row["id"]) == memory_id
        assert str(postgres_row["user_id"]) == user_id
        assert postgres_row["content"] == content
        print("PostgreSQL memory ID:", memory_id)

        print("\n[2] Qdrant UPSERT + VERIFY")
        assert qdrant.upsert_memory(
            memory_id=memory_id,
            text=content,
            user_id=user_id,
            metadata={
                "importance": importance,
                "confidence": confidence,
                "emotions": emotions,
                "evidence_ids": evidence_ids,
                "is_active": True,
            },
        )

        # Direct retrieval prevents an in-memory fallback from creating a false pass.
        qdrant_points = qdrant.client.retrieve(
            collection_name=qdrant.COLLECTION_NAME,
            ids=[memory_id],
            with_payload=True,
            with_vectors=False,
        )
        assert len(qdrant_points) == 1
        assert str(qdrant_points[0].id) == memory_id
        assert qdrant_points[0].payload["memory_id"] == memory_id
        assert qdrant_points[0].payload["user_id"] == user_id
        assert qdrant_points[0].payload["evidence_ids"] == evidence_ids

        search_results = qdrant.search_long_term_memories(
            user_id=user_id,
            query_text=content,
            limit=5,
            score_threshold=0.0,
        )
        matching_results = [
            result
            for result in search_results
            if result.get("memory_id") == memory_id
        ]
        assert matching_results
        print("Qdrant memory ID:", matching_results[0]["memory_id"])

        print("\n[3] Neo4j UPSERT, LINK + VERIFY")
        assert neo4j.upsert_memory_node(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            importance=importance,
            confidence=confidence,
        )
        assert neo4j.link_student_memory(user_id=user_id, memory_id=memory_id)

        # An empty relationship list does not prove that a Memory node exists.
        with neo4j._driver.session() as session:
            neo4j_row = session.run(
                """
                MATCH (student:Student {id: $user_id})-[:HAS_MEMORY]->
                      (memory:Memory {id: $memory_id, user_id: $user_id})
                RETURN memory.id AS memory_id,
                       memory.content AS content,
                       memory.importance AS importance,
                       memory.confidence AS confidence
                """,
                user_id=user_id,
                memory_id=memory_id,
            ).single()
        assert neo4j_row is not None
        assert neo4j_row["memory_id"] == memory_id
        assert neo4j_row["content"] == content
        print("Neo4j memory ID:", neo4j_row["memory_id"])

        print("\n[4] UUID CONSISTENCY")
        assert str(postgres_row["id"]) == memory_id
        assert qdrant_points[0].payload["memory_id"] == memory_id
        assert neo4j_row["memory_id"] == memory_id
        print("Same memory UUID verified across all three real systems.")

    finally:
        # These adapters do not currently expose public deletion for every store,
        # so cleanup uses their live connections directly where required.
        if memory_id is not None:
            if qdrant.client is not None:
                qdrant.delete_memory(memory_id)

            if neo4j._driver is not None:
                with neo4j._driver.session() as session:
                    session.run(
                        "MATCH (memory:Memory {id: $memory_id}) DETACH DELETE memory",
                        memory_id=memory_id,
                    ).consume()
                    session.run(
                        """
                        MATCH (student:Student {id: $user_id})
                        WHERE NOT (student)--()
                        DELETE student
                        """,
                        user_id=user_id,
                    ).consume()

            if database.is_postgres_connected and database._pool is not None:
                async with database._pool.acquire() as connection:
                    await connection.execute(
                        "DELETE FROM long_term_memories WHERE id = $1::uuid",
                        uuid.UUID(memory_id),
                    )

        neo4j.close()
        await database.close()
        print("\nTest data cleaned up.")


@pytest.mark.asyncio
async def test_real_stage1_relationship_maps_to_final_memory_relationship() -> None:
    """Prove group_A -> group_B becomes final Memory A -> Memory B in Neo4j."""

    user_id = str(uuid.uuid4())
    database = DatabaseManager()
    qdrant = QdrantAdapter()
    neo4j = Neo4jAdapter()
    memory_ids = []

    candidates = [
        CandidateMemory(
            id="integration-rel-cm-a",
            user_id=user_id,
            content="Studying alone for long periods makes me feel isolated.",
            importance=0.82,
            confidence=0.90,
        ),
        CandidateMemory(
            id="integration-rel-cm-b",
            user_id=user_id,
            content="After that isolation, I try to find someone trustworthy to talk to.",
            importance=0.86,
            confidence=0.91,
        ),
    ]

    try:
        await database.initialize()
        assert database.is_postgres_connected and database._pool is not None, (
            "Neon PostgreSQL is not connected. Check DATABASE_URL and network access."
        )
        assert qdrant.client is not None, "Qdrant is not configured or connected."
        assert neo4j._driver is not None, "Neo4j is not configured or connected."
        qdrant.client.get_collection(qdrant.COLLECTION_NAME)
        neo4j._driver.verify_connectivity()

        engine = MemoryEngine(
            database=database,
            qdrant=qdrant,
            neo4j=neo4j,
            data_agent_instance=TwoMemoryCreateAgent(),
        )

        print("\n========== REAL RELATIONSHIP PIPELINE TEST ==========")
        print("\n[STAGE 1]")
        print("integration-rel-cm-a -> integration-group-a")
        print("integration-rel-cm-b -> integration-group-b")
        print("integration-group-a --TRIGGERS--> integration-group-b")

        result = await engine.process_long_term_memories(
            candidates,
            score_threshold=0.0,
        )

        memory_a_id = result.consolidation_memory_map["integration-group-a"]
        memory_b_id = result.consolidation_memory_map["integration-group-b"]
        memory_ids = [memory_a_id, memory_b_id]
        assert memory_a_id != memory_b_id
        assert len(result.persistence) == 2
        assert all(item.postgres_operation == "CREATE" for item in result.persistence)
        assert all(item.qdrant_operation == "UPSERT" for item in result.persistence)
        assert len(result.graph_relationships) == 1
        assert (
            result.graph_relationships[0].status
            == GraphRelationshipStatus.CREATED
        )
        assert result.graph_relationships[0].source_memory_id == memory_a_id
        assert result.graph_relationships[0].target_memory_id == memory_b_id

        print("\n[STAGE 2 -> FINAL UUID MAPPING]")
        print(f"integration-group-a -> Memory A: {memory_a_id}")
        print(f"integration-group-b -> Memory B: {memory_b_id}")

        print("\n[POSTGRESQL VERIFY]")
        async with database._pool.acquire() as connection:
            postgres_rows = await connection.fetch(
                """
                SELECT id, user_id, content
                FROM long_term_memories
                WHERE user_id = $1::uuid AND id = ANY($2::uuid[])
                """,
                uuid.UUID(user_id),
                [uuid.UUID(item) for item in memory_ids],
            )
        postgres_by_id = {str(row["id"]): dict(row) for row in postgres_rows}
        assert set(postgres_by_id) == set(memory_ids)
        print(f"Memory A stored: {memory_a_id}")
        print(f"Memory B stored: {memory_b_id}")

        print("\n[QDRANT VERIFY]")
        qdrant_points = qdrant.client.retrieve(
            collection_name=qdrant.COLLECTION_NAME,
            ids=memory_ids,
            with_payload=True,
            with_vectors=False,
        )
        qdrant_by_id = {str(point.id): point for point in qdrant_points}
        assert set(qdrant_by_id) == set(memory_ids)
        assert all(
            qdrant_by_id[item].payload["user_id"] == user_id
            for item in memory_ids
        )
        print(f"Memory A stored: {memory_a_id}")
        print(f"Memory B stored: {memory_b_id}")

        print("\n[NEO4J VERIFY]")
        with neo4j._driver.session() as session:
            graph_row = session.run(
                """
                MATCH (source:Memory {
                    id: $source_memory_id,
                    user_id: $user_id
                })-[relationship:TRIGGERS]->(target:Memory {
                    id: $target_memory_id,
                    user_id: $user_id
                })
                RETURN source.id AS source_memory_id,
                       target.id AS target_memory_id,
                       relationship.evidence_ids AS evidence_ids,
                       relationship.confidence AS confidence
                """,
                user_id=user_id,
                source_memory_id=memory_a_id,
                target_memory_id=memory_b_id,
            ).single()

        assert graph_row is not None
        assert graph_row["source_memory_id"] == memory_a_id
        assert graph_row["target_memory_id"] == memory_b_id
        assert graph_row["evidence_ids"] == [
            "integration-rel-cm-a",
            "integration-rel-cm-b",
        ]
        assert graph_row["confidence"] == pytest.approx(0.87)

        print("\nFINAL NEO4J RELATIONSHIP:")
        print(f"Memory A ({memory_a_id})")
        print("        |")
        print("        | TRIGGERS")
        print("        v")
        print(f"Memory B ({memory_b_id})")
        print("\nPostgreSQL: Memory A + Memory B  [VERIFIED]")
        print("Qdrant:    Memory A + Memory B  [VERIFIED]")
        print("Neo4j:     Memory A --TRIGGERS--> Memory B  [VERIFIED]")
        print("========== RELATIONSHIP PIPELINE PASSED ==========")

    finally:
        # Query by the unique test user as well, so partial pipeline failures are
        # still cleaned even if the result mapping was not returned.
        if database.is_postgres_connected and database._pool is not None:
            async with database._pool.acquire() as connection:
                stored_ids = await connection.fetch(
                    "SELECT id FROM long_term_memories WHERE user_id = $1::uuid",
                    uuid.UUID(user_id),
                )
                memory_ids = list(
                    dict.fromkeys(memory_ids + [str(row["id"]) for row in stored_ids])
                )

        if qdrant.client is not None:
            for memory_id in memory_ids:
                qdrant.delete_memory(memory_id)

        if neo4j._driver is not None:
            with neo4j._driver.session() as session:
                session.run(
                    "MATCH (memory:Memory {user_id: $user_id}) DETACH DELETE memory",
                    user_id=user_id,
                ).consume()
                session.run(
                    """
                    MATCH (student:Student {id: $user_id})
                    WHERE NOT (student)--()
                    DELETE student
                    """,
                    user_id=user_id,
                ).consume()

        if database.is_postgres_connected and database._pool is not None:
            async with database._pool.acquire() as connection:
                await connection.execute(
                    "DELETE FROM long_term_memories WHERE user_id = $1::uuid",
                    uuid.UUID(user_id),
                )

        neo4j.close()
        await database.close()
        print("\nRelationship test data safely cleaned from all three stores.")
