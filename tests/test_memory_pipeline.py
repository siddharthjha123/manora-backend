"""Deterministic tests for two-stage memory orchestration and persistence."""

import json
import uuid

import pytest

from data_agent.data_engine import DataAgentValidationError
from data_agent.data_schema import (
    CandidateMemory,
    ConsolidationRelationship,
    ConsolidatedMemory,
    GraphRelationshipStatus,
    MemoryDecision,
    MemoryDecisionType,
    Stage1ConsolidationResult,
)
from memory.memory_engine import MemoryEngine


USER_ID = "550e8400-e29b-41d4-a716-446655440000"
OTHER_USER_ID = "650e8400-e29b-41d4-a716-446655440000"
EXISTING_MEMORY_ID = "0b5bea47-39a5-4fa9-abc0-9365ae764b0b"
SECOND_MEMORY_ID = "1b5bea47-39a5-4fa9-abc0-9365ae764b0b"
CREATED_MEMORY_A = "2b5bea47-39a5-4fa9-abc0-9365ae764b0b"
CREATED_MEMORY_B = "3b5bea47-39a5-4fa9-abc0-9365ae764b0b"


def candidate(memory_id: str, content: str) -> CandidateMemory:
    return CandidateMemory(
        id=memory_id,
        user_id=USER_ID,
        content=content,
        importance=0.84,
        confidence=0.89,
    )


def consolidated(
    candidate_ids,
    content,
    *,
    consolidation_id="group-1",
    emotions=None,
) -> ConsolidatedMemory:
    return ConsolidatedMemory(
        consolidation_id=consolidation_id,
        candidate_ids=candidate_ids,
        evidence_ids=candidate_ids,
        content=content,
        emotions=(
            [{"emotion": "loneliness", "confidence": 0.91}]
            if emotions is None
            else emotions
        ),
        importance=0.84,
        confidence=0.89,
    )


class FakeDataAgent:
    def __init__(self, consolidated_memory, decision, relationships=None):
        self.consolidated_memories = (
            consolidated_memory
            if isinstance(consolidated_memory, list)
            else [consolidated_memory]
        )
        self.decisions = (
            decision
            if isinstance(decision, dict)
            else {self.consolidated_memories[0].consolidation_id: decision}
        )
        self.relationships = relationships or []
        self.stage2_existing_memories = None

    async def consolidate_candidates(self, candidate_memories):
        return Stage1ConsolidationResult(
            user_id=USER_ID,
            consolidated_memories=self.consolidated_memories,
            relationships=self.relationships,
            reasoning_summary="The mock Stage 1 response consolidated this fixture.",
        )

    async def decide_memory_actions(
        self,
        *,
        user_id,
        consolidated_memory,
        existing_long_term_memories,
    ):
        self.stage2_existing_memories = existing_long_term_memories
        return self.decisions[consolidated_memory.consolidation_id]


class FakeDatabase:
    def __init__(self, existing_memory=None, existing_memories=None, create_ids=None):
        self.created = []
        self.updated = []
        self.existing_memory = existing_memory
        self.existing_memories = {
            str(memory["id"]): memory for memory in (existing_memories or [])
        }
        if existing_memory:
            self.existing_memories[str(existing_memory["id"])] = existing_memory
        self.create_ids = list(create_ids or [])

    async def get_long_term_memory(self, **values):
        existing_memory = self.existing_memories.get(str(values["memory_id"]))
        if not existing_memory:
            return None
        if str(existing_memory["user_id"]) != str(values["user_id"]):
            return None
        return existing_memory

    async def create_long_term_memory(self, **values):
        self.created.append(values)
        return {
            "id": self.create_ids.pop(0) if self.create_ids else str(uuid.uuid4()),
            "user_id": values["user_id"],
            **values,
            "is_active": True,
        }

    async def update_long_term_memory(self, **values):
        self.updated.append(values)
        return {
            "id": values["memory_id"],
            "user_id": values["user_id"],
            **values,
            "is_active": True,
        }


class FakeQdrant:
    def __init__(self, search_results=None):
        self.search_results = search_results or []
        self.search_calls = []
        self.upserts = []

    def search_long_term_memories(self, **values):
        self.search_calls.append(values)
        if isinstance(self.search_results, dict):
            return self.search_results.get(values["query_text"], [])
        return self.search_results

    def upsert_memory(self, **values):
        self.upserts.append(values)
        return True


class FakeNeo4j:
    def __init__(self, *, relationship_result=True):
        self.nodes = []
        self.links = []
        self.relationships = []
        self.relationship_result = relationship_result

    def upsert_memory_node(self, **values):
        self.nodes.append(values)
        return True

    def link_student_memory(self, **values):
        self.links.append(values)
        return True

    def create_memory_relationship(self, **values):
        self.relationships.append(values)
        return self.relationship_result


def decision(
    action,
    candidate_ids,
    *,
    memory_id=None,
    evidence_ids=None,
    content=None,
    emotions=None,
    reasoning="The retrieved memories support this decision.",
):
    return MemoryDecision(
        action=action,
        memory_id=memory_id,
        candidate_ids=candidate_ids,
        evidence_ids=evidence_ids or candidate_ids,
        content=content,
        emotions=(
            [{"emotion": "loneliness", "confidence": 0.91}]
            if emotions is None
            else emotions
        ),
        importance=0.86,
        confidence=0.91,
        reasoning=reasoning,
    )


def existing_search_result(*, user_id=USER_ID):
    return {
        "memory_id": EXISTING_MEMORY_ID,
        "text": "Student struggles to form close friendships.",
        "score": 0.7,
        "metadata": {
            "memory_id": EXISTING_MEMORY_ID,
            "user_id": user_id,
            "importance": 0.84,
            "confidence": 0.89,
            "emotions": [{"emotion": "loneliness", "confidence": 0.91}],
            "evidence_ids": ["cm-old"],
            "is_active": True,
        },
    }


def print_pipeline(label, result):
    print(f"\n========== {label} RESPONSE ==========")
    print(json.dumps(result.model_dump(mode="json"), indent=2, default=str))


@pytest.mark.asyncio
async def test_stage2_create_persists_postgres_then_same_id_to_qdrant():
    candidate_ids = ["cm-create-1"]
    consolidated_memory = consolidated(
        candidate_ids,
        "Student wants to build a consistent exercise routine.",
        consolidation_id="group-create",
        emotions=[{"emotion": "motivation", "confidence": 0.86}],
    )
    database = FakeDatabase()
    qdrant = FakeQdrant()
    fake_agent = FakeDataAgent(
        consolidated_memory,
        decision(
            MemoryDecisionType.CREATE,
            candidate_ids,
            content=consolidated_memory.content,
            emotions=[{"emotion": "motivation", "confidence": 0.86}],
            reasoning="No related Qdrant memory was retrieved, so create a new memory.",
        ),
    )
    engine = MemoryEngine(
        database=database,
        qdrant=qdrant,
        neo4j=FakeNeo4j(),
        data_agent_instance=fake_agent,
    )

    result = await engine.process_long_term_memories(
        [candidate("cm-create-1", "I want to exercise consistently this semester.")]
    )
    print_pipeline("CREATE", result)

    assert result.decisions[0].action == MemoryDecisionType.CREATE
    assert result.decisions[0].memory_id is None
    assert len(database.created) == 1
    assert len(qdrant.upserts) == 1
    assert result.persistence[0].memory_id == qdrant.upserts[0]["memory_id"]
    assert database.created[0]["evidence_ids"] == ["cm-create-1"]
    assert qdrant.upserts[0]["metadata"]["evidence_ids"] == ["cm-create-1"]
    assert database.updated == []


@pytest.mark.asyncio
async def test_stage2_update_reuses_postgres_and_qdrant_memory_id():
    candidate_ids = ["cm-new-23"]
    consolidated_memory = consolidated(
        candidate_ids,
        "Student still feels lonely and wants someone trustworthy to talk to.",
        consolidation_id="group-update",
    )
    database = FakeDatabase(
        existing_memory={
            "id": EXISTING_MEMORY_ID,
            "user_id": USER_ID,
            "content": "Student struggles to form close friendships.",
            "evidence_ids": '["cm-old-01", "cm-old-02"]',
            "is_active": True,
        }
    )
    qdrant = FakeQdrant([existing_search_result()])
    fake_agent = FakeDataAgent(
        consolidated_memory,
        decision(
            MemoryDecisionType.UPDATE,
            candidate_ids,
            memory_id=EXISTING_MEMORY_ID,
            # Stage 2 supplies only the new evidence. Persistence must retain old IDs.
            evidence_ids=["cm-new-23"],
            content="Student experiences loneliness and strongly wants close friendships.",
            reasoning="The retrieved memory represents the same persistent social pattern.",
        ),
    )
    engine = MemoryEngine(
        database=database,
        qdrant=qdrant,
        neo4j=FakeNeo4j(),
        data_agent_instance=fake_agent,
    )

    result = await engine.process_long_term_memories(
        [candidate("cm-new-23", "I still have nobody I can genuinely talk to.")]
    )
    print_pipeline("UPDATE", result)

    assert result.decisions[0].action == MemoryDecisionType.UPDATE
    assert database.created == []
    assert len(database.updated) == 1
    assert database.updated[0]["memory_id"] == EXISTING_MEMORY_ID
    assert database.updated[0]["evidence_ids"] == [
        "cm-old-01",
        "cm-old-02",
        "cm-new-23",
    ]
    assert qdrant.upserts[0]["memory_id"] == EXISTING_MEMORY_ID
    assert qdrant.upserts[0]["metadata"]["evidence_ids"] == [
        "cm-old-01",
        "cm-old-02",
        "cm-new-23",
    ]
    assert result.persistence[0].memory_id == EXISTING_MEMORY_ID


@pytest.mark.asyncio
async def test_stage2_reject_performs_no_persistence():
    candidate_ids = ["cm-temp-1"]
    consolidated_memory = consolidated(
        candidate_ids,
        "Student is going to get lunch right now.",
        consolidation_id="group-reject",
        emotions=[],
    )
    database = FakeDatabase()
    qdrant = FakeQdrant()
    fake_agent = FakeDataAgent(
        consolidated_memory,
        decision(
            MemoryDecisionType.REJECT,
            candidate_ids,
            content=None,
            emotions=[],
            reasoning="Going to lunch is temporary and unsuitable for long-term memory.",
        ),
    )
    engine = MemoryEngine(
        database=database,
        qdrant=qdrant,
        neo4j=FakeNeo4j(),
        data_agent_instance=fake_agent,
    )

    result = await engine.process_long_term_memories(
        [candidate("cm-temp-1", "I am going to get lunch now.")]
    )
    print_pipeline("REJECT", result)

    assert result.persistence[0].postgres_operation == "NONE"
    assert result.persistence[0].qdrant_operation == "NONE"
    assert database.created == []
    assert database.updated == []
    assert qdrant.upserts == []


@pytest.mark.asyncio
async def test_cross_user_qdrant_result_is_never_passed_to_stage2():
    candidate_ids = ["cm-isolation-1"]
    consolidated_memory = consolidated(
        candidate_ids,
        "Student wants a consistent exercise routine.",
        consolidation_id="group-isolation",
        emotions=[{"emotion": "motivation", "confidence": 0.86}],
    )
    database = FakeDatabase()
    qdrant = FakeQdrant([existing_search_result(user_id=OTHER_USER_ID)])
    fake_agent = FakeDataAgent(
        consolidated_memory,
        decision(
            MemoryDecisionType.CREATE,
            candidate_ids,
            content=consolidated_memory.content,
            emotions=[{"emotion": "motivation", "confidence": 0.86}],
        ),
    )
    engine = MemoryEngine(
        database=database,
        qdrant=qdrant,
        neo4j=FakeNeo4j(),
        data_agent_instance=fake_agent,
    )

    await engine.process_long_term_memories(
        [candidate("cm-isolation-1", "I want to exercise consistently.")]
    )

    assert fake_agent.stage2_existing_memories == []


def relationship(source="group-a", target="group-b"):
    return ConsolidationRelationship(
        source_id=source,
        relation="TRIGGERS",
        target_id=target,
        evidence_ids=["cm-a", "cm-b"],
        confidence=0.87,
    )


def two_group_fixture(action_a, action_b, *, database, qdrant=None, neo4j=None):
    group_a = consolidated(["cm-a"], "Persistent memory A", consolidation_id="group-a")
    group_b = consolidated(["cm-b"], "Persistent memory B", consolidation_id="group-b")
    agent = FakeDataAgent(
        [group_a, group_b],
        {"group-a": action_a, "group-b": action_b},
        [relationship()],
    )
    graph = neo4j or FakeNeo4j()
    engine = MemoryEngine(
        database=database,
        qdrant=qdrant or FakeQdrant(),
        neo4j=graph,
        data_agent_instance=agent,
    )
    return engine, graph


@pytest.mark.asyncio
async def test_create_create_relationship_uses_only_final_memory_ids():
    database = FakeDatabase(create_ids=[CREATED_MEMORY_A, CREATED_MEMORY_B])
    engine, graph = two_group_fixture(
        decision(MemoryDecisionType.CREATE, ["cm-a"], content="Persistent memory A"),
        decision(MemoryDecisionType.CREATE, ["cm-b"], content="Persistent memory B"),
        database=database,
    )

    result = await engine.process_long_term_memories(
        [candidate("cm-a", "Observation A"), candidate("cm-b", "Observation B")]
    )

    assert result.consolidation_memory_map == {
        "group-a": CREATED_MEMORY_A,
        "group-b": CREATED_MEMORY_B,
    }
    assert graph.relationships[0]["source_memory_id"] == CREATED_MEMORY_A
    assert graph.relationships[0]["target_memory_id"] == CREATED_MEMORY_B
    assert result.graph_relationships[0].status == GraphRelationshipStatus.CREATED


@pytest.mark.asyncio
async def test_update_update_relationship_reuses_both_existing_memory_ids():
    existing = [
        {"id": EXISTING_MEMORY_ID, "user_id": USER_ID, "evidence_ids": ["old-a"]},
        {"id": SECOND_MEMORY_ID, "user_id": USER_ID, "evidence_ids": ["old-b"]},
    ]
    database = FakeDatabase(existing_memories=existing)
    engine, graph = two_group_fixture(
        decision(MemoryDecisionType.UPDATE, ["cm-a"], memory_id=EXISTING_MEMORY_ID, content="Updated A"),
        decision(MemoryDecisionType.UPDATE, ["cm-b"], memory_id=SECOND_MEMORY_ID, content="Updated B"),
        database=database,
    )

    result = await engine.process_long_term_memories(
        [candidate("cm-a", "Observation A"), candidate("cm-b", "Observation B")]
    )

    assert result.consolidation_memory_map == {
        "group-a": EXISTING_MEMORY_ID,
        "group-b": SECOND_MEMORY_ID,
    }
    assert graph.relationships[0]["source_memory_id"] == EXISTING_MEMORY_ID
    assert graph.relationships[0]["target_memory_id"] == SECOND_MEMORY_ID


@pytest.mark.asyncio
async def test_create_reject_relationship_is_skipped():
    database = FakeDatabase(create_ids=[CREATED_MEMORY_A])
    engine, graph = two_group_fixture(
        decision(MemoryDecisionType.CREATE, ["cm-a"], content="Persistent memory A"),
        decision(MemoryDecisionType.REJECT, ["cm-b"], content=None),
        database=database,
    )

    result = await engine.process_long_term_memories(
        [candidate("cm-a", "Observation A"), candidate("cm-b", "Observation B")]
    )

    assert result.consolidation_memory_map == {"group-a": CREATED_MEMORY_A}
    assert graph.relationships == []
    assert result.graph_relationships[0].status == GraphRelationshipStatus.SKIPPED_MISSING_ENDPOINT
    assert result.graph_relationships[0].target_memory_id is None


@pytest.mark.asyncio
async def test_update_create_relationship_translates_mixed_actions():
    database = FakeDatabase(
        existing_memory={
            "id": EXISTING_MEMORY_ID,
            "user_id": USER_ID,
            "evidence_ids": ["old-a"],
        },
        create_ids=[CREATED_MEMORY_B],
    )
    engine, graph = two_group_fixture(
        decision(MemoryDecisionType.UPDATE, ["cm-a"], memory_id=EXISTING_MEMORY_ID, content="Updated A"),
        decision(MemoryDecisionType.CREATE, ["cm-b"], content="Persistent memory B"),
        database=database,
    )

    result = await engine.process_long_term_memories(
        [candidate("cm-a", "Observation A"), candidate("cm-b", "Observation B")]
    )

    assert result.consolidation_memory_map == {
        "group-a": EXISTING_MEMORY_ID,
        "group-b": CREATED_MEMORY_B,
    }
    assert graph.relationships[0]["source_memory_id"] == EXISTING_MEMORY_ID
    assert graph.relationships[0]["target_memory_id"] == CREATED_MEMORY_B


@pytest.mark.asyncio
async def test_candidate_id_relationship_endpoint_is_rejected_before_persistence():
    group_a = consolidated(["cm-a"], "Persistent memory A", consolidation_id="group-a")
    group_b = consolidated(["cm-b"], "Persistent memory B", consolidation_id="group-b")
    agent = FakeDataAgent(
        [group_a, group_b],
        {
            "group-a": decision(MemoryDecisionType.CREATE, ["cm-a"], content="A"),
            "group-b": decision(MemoryDecisionType.CREATE, ["cm-b"], content="B"),
        },
        [relationship(source="cm-a")],
    )
    database = FakeDatabase()
    qdrant = FakeQdrant()
    graph = FakeNeo4j()
    engine = MemoryEngine(database=database, qdrant=qdrant, neo4j=graph, data_agent_instance=agent)

    with pytest.raises(DataAgentValidationError, match="never candidate memory IDs"):
        await engine.process_long_term_memories(
            [candidate("cm-a", "Observation A"), candidate("cm-b", "Observation B")]
        )

    assert database.created == []
    assert qdrant.search_calls == []
    assert graph.nodes == []


@pytest.mark.asyncio
async def test_cross_user_graph_rejection_is_reported_by_pipeline():
    database = FakeDatabase(create_ids=[CREATED_MEMORY_A, CREATED_MEMORY_B])
    graph = FakeNeo4j(relationship_result=False)
    engine, graph = two_group_fixture(
        decision(MemoryDecisionType.CREATE, ["cm-a"], content="Persistent memory A"),
        decision(MemoryDecisionType.CREATE, ["cm-b"], content="Persistent memory B"),
        database=database,
        neo4j=graph,
    )

    result = await engine.process_long_term_memories(
        [candidate("cm-a", "Observation A"), candidate("cm-b", "Observation B")]
    )

    assert result.graph_relationships[0].status == GraphRelationshipStatus.REJECTED
    assert graph.relationships[0]["user_id"] == USER_ID
