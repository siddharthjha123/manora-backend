"""Deterministic unit tests for the LLM-driven Data Agent V1."""

import json

import pytest

from data_agent.data_engine import DataAgent, DataAgentValidationError
from data_agent.data_llm import DataAgentLLM
from data_agent.data_schema import (
    CandidateMemory,
    CandidateMemoryEmotion,
    ExistingLongTermMemory,
    MemoryActionType,
)
from llm.base import LLMClient, LLMJSONParseError


class MockLLMClient(LLMClient):
    """Return one controlled raw response while exercising real JSON validation."""

    def __init__(self, raw_response: str):
        self.raw_response = raw_response
        self.call_count = 0
        self.last_messages = None

    async def generate(self, messages, **kwargs):
        self.call_count += 1
        self.last_messages = messages
        return self.raw_response


def candidate(
    memory_id: str,
    content: str,
    *,
    user_id: str = "user-1",
    emotions=None,
) -> CandidateMemory:
    return CandidateMemory(
        id=memory_id,
        user_id=user_id,
        content=content,
        emotional_state=emotions or [],
        importance=0.8,
        confidence=0.9,
    )


def existing_memory(
    memory_id: str,
    content: str,
    *,
    user_id: str = "user-1",
) -> ExistingLongTermMemory:
    return ExistingLongTermMemory(
        id=memory_id,
        user_id=user_id,
        content=content,
        evidence_ids=["cm-old"],
        importance=0.8,
        confidence=0.9,
    )


def action(
    action_type: str,
    candidate_ids,
    *,
    action_id: str = "action-1",
    memory_id=None,
    content=None,
    evidence_ids=None,
    emotions=None,
):
    return {
        "action_id": action_id,
        "action": action_type,
        "memory_id": memory_id,
        "candidate_ids": candidate_ids,
        "evidence_ids": evidence_ids or candidate_ids,
        "content": content,
        "emotions": emotions or [],
        "importance": 0.82,
        "confidence": 0.9,
        "reasoning": "The supplied evidence supports this memory decision.",
    }


def result_json(actions, *, relationships=None, user_id="user-1"):
    return json.dumps(
        {
            "user_id": user_id,
            "memory_actions": actions,
            "relationships": relationships or [],
            "reasoning_summary": "All candidates were evaluated as one student batch.",
        }
    )


def data_agent(raw_response: str):
    client = MockLLMClient(raw_response)
    agent = DataAgent(reasoning_agent=DataAgentLLM(client=client))
    return agent, client


@pytest.mark.asyncio
async def test_empty_input_returns_empty_result_without_calling_llm():
    agent, client = data_agent("not used")

    result = await agent.consolidate([])

    assert result.memory_actions == []
    assert result.relationships == []
    assert client.call_count == 0


@pytest.mark.asyncio
async def test_single_meaningful_candidate_creates_memory():
    response = result_json([
        action(
            "CREATE",
            ["cm-1"],
            content="Student wants to pursue machine learning as a career direction.",
        )
    ])
    agent, client = data_agent(response)

    result = await agent.consolidate([
        candidate("cm-1", "I want to build a career in machine learning.")
    ])

    assert result.memory_actions[0].action == MemoryActionType.CREATE
    assert result.memory_actions[0].evidence_ids == ["cm-1"]
    assert "Memory Consolidation Agent" in client.last_messages[0]["content"]


@pytest.mark.asyncio
async def test_related_new_candidates_merge_into_one_memory():
    candidate_ids = ["cm-1", "cm-2", "cm-3"]
    response = result_json([
        action(
            "MERGE",
            candidate_ids,
            content=(
                "Student experiences loneliness, lacks close support, and strongly "
                "wants genuine friendships."
            ),
        )
    ])
    agent, _ = data_agent(response)

    result = await agent.consolidate([
        candidate("cm-1", "I feel lonely at college."),
        candidate("cm-2", "I do not have anyone I can call."),
        candidate("cm-3", "I want genuine friends."),
    ])

    merge = result.memory_actions[0]
    assert merge.action == MemoryActionType.MERGE
    assert merge.evidence_ids == candidate_ids


@pytest.mark.asyncio
async def test_existing_memory_is_updated_instead_of_duplicated():
    response = result_json([
        action(
            "UPDATE",
            ["cm-19", "cm-20"],
            memory_id="mem-1",
            evidence_ids=["cm-old", "cm-19", "cm-20"],
            content=(
                "Student struggles to form close relationships while strongly "
                "wanting genuine social connection."
            ),
        )
    ])
    agent, _ = data_agent(response)

    result = await agent.consolidate(
        [
            candidate("cm-19", "I do not know how to make close friends."),
            candidate("cm-20", "I want someone I can genuinely talk to."),
        ],
        [existing_memory("mem-1", "Student struggles to form close relationships.")],
    )

    update = result.memory_actions[0]
    assert update.action == MemoryActionType.UPDATE
    assert update.memory_id == "mem-1"
    assert len(result.memory_actions) == 1


@pytest.mark.asyncio
async def test_unrelated_candidate_creates_with_existing_memories_present():
    response = result_json([
        action(
            "CREATE",
            ["cm-1"],
            content="Student wants to pursue machine learning professionally.",
        )
    ])
    agent, _ = data_agent(response)

    result = await agent.consolidate(
        [candidate("cm-1", "I want a career in machine learning.")],
        [existing_memory("mem-social", "Student struggles to form friendships.")],
    )

    assert result.memory_actions[0].action == MemoryActionType.CREATE
    assert result.memory_actions[0].memory_id is None


@pytest.mark.asyncio
async def test_trivial_candidate_can_be_rejected_by_llm():
    response = result_json([
        action("REJECT", ["cm-1"], content=None)
    ])
    agent, _ = data_agent(response)

    result = await agent.consolidate([
        candidate("cm-1", "I am going to get lunch now.")
    ])

    assert result.memory_actions[0].action == MemoryActionType.REJECT
    assert result.memory_actions[0].content is None


@pytest.mark.asyncio
async def test_relationship_discovery_preserves_evidence():
    response = result_json(
        [
            action(
                "MERGE",
                ["cm-13", "cm-14"],
                content="A past friendship loss contributes to difficulty trusting people.",
            )
        ],
        relationships=[
            {
                "source_id": "cm-13",
                "source_type": "candidate_memory",
                "relation": "INFLUENCES",
                "target_id": "action-1",
                "target_type": "memory_action",
                "evidence_ids": ["cm-13", "cm-14"],
                "confidence": 0.86,
            }
        ],
    )
    agent, _ = data_agent(response)

    result = await agent.consolidate([
        candidate("cm-13", "A close school friendship became distant."),
        candidate("cm-14", "I now hesitate to trust new people."),
    ])

    assert result.relationships[0].relation.value == "INFLUENCES"
    assert result.relationships[0].evidence_ids == ["cm-13", "cm-14"]


@pytest.mark.asyncio
async def test_invented_evidence_id_is_rejected():
    response = result_json([
        action(
            "CREATE",
            ["cm-1"],
            evidence_ids=["cm-1", "cm-invented"],
            content="Student has a persistent career goal.",
        )
    ])
    agent, _ = data_agent(response)

    with pytest.raises(DataAgentValidationError, match="invented evidence IDs"):
        await agent.consolidate([candidate("cm-1", "I want a meaningful job.")])


@pytest.mark.asyncio
async def test_unknown_update_memory_id_is_rejected():
    response = result_json([
        action(
            "UPDATE",
            ["cm-1"],
            memory_id="mem-999",
            content="Student wants stronger social support.",
        )
    ])
    agent, _ = data_agent(response)

    with pytest.raises(DataAgentValidationError, match="unknown memory_id"):
        await agent.consolidate(
            [candidate("cm-1", "I want someone to talk to.")],
            [existing_memory("mem-1", "Student feels socially isolated.")],
        )


@pytest.mark.asyncio
async def test_mixed_user_input_is_refused_before_llm_call():
    agent, client = data_agent("not used")

    with pytest.raises(DataAgentValidationError, match="same non-empty user_id"):
        await agent.consolidate([
            candidate("cm-a", "I feel lonely.", user_id="user-a"),
            candidate("cm-b", "I feel lonely.", user_id="user-b"),
        ])

    assert client.call_count == 0


@pytest.mark.asyncio
async def test_malformed_llm_output_raises_parse_error():
    agent, _ = data_agent("this is not JSON")

    with pytest.raises(LLMJSONParseError):
        await agent.consolidate([candidate("cm-1", "I value genuine friendships.")])


@pytest.mark.asyncio
async def test_multiple_emotions_are_preserved():
    emotions = [
        {"emotion": "anxiety", "confidence": 0.84},
        {"emotion": "frustration", "confidence": 0.71},
        {"emotion": "sadness", "confidence": 0.22},
    ]
    response = result_json([
        action(
            "CREATE",
            ["cm-1"],
            content="Student experiences layered emotions about placement progress.",
            emotions=emotions,
        )
    ])
    agent, _ = data_agent(response)

    result = await agent.consolidate([
        candidate(
            "cm-1",
            "Placement uncertainty makes me frustrated, anxious, and sad.",
            emotions=[
                CandidateMemoryEmotion(emotion=item["emotion"], confidence=item["confidence"])
                for item in emotions
            ],
        )
    ])

    assert [emotion.emotion for emotion in result.memory_actions[0].emotions] == [
        "anxiety",
        "frustration",
        "sadness",
    ]
