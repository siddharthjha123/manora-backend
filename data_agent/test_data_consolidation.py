"""Focused tests for the isolated deterministic Data Agent V1."""

import pytest

from data_agent.data_engine import DataAgent
from data_agent.data_schema import CandidateMemory, CandidateMemoryEmotion


def candidate(
    memory_id: str,
    *,
    user_id: str = "user-1",
    content: str = "Student feels lonely and wants a close friend to talk to.",
    subtopic: str = "loneliness",
    importance: float = 0.82,
    confidence: float = 0.88,
    emotions=None,
) -> CandidateMemory:
    return CandidateMemory(
        id=memory_id,
        user_id=user_id,
        content=content,
        context={"domain": "social", "subtopic": subtopic},
        emotional_state=emotions or [],
        importance=importance,
        confidence=confidence,
    )


def test_empty_input():
    result = DataAgent().consolidate([])

    assert result.promoted_memories == []
    assert result.patterns == []
    assert result.relationships == []
    assert result.rejected_memory_ids == []


def test_single_candidate_is_promoted():
    result = DataAgent().consolidate([candidate("cm-1")])

    assert len(result.promoted_memories) == 1
    assert result.promoted_memories[0].evidence_ids == ["cm-1"]
    assert result.promoted_memories[0].support_count == 1
    assert result.patterns == []


def test_duplicate_candidates_are_consolidated_once():
    first = candidate("cm-1", content="Student often feels lonely at college.")
    duplicate = candidate("cm-2", content="Student often feels lonely at college.")

    result = DataAgent().consolidate([first, duplicate])

    assert len(result.promoted_memories) == 1
    assert result.promoted_memories[0].evidence_ids == ["cm-1", "cm-2"]
    assert len(result.patterns) == 1
    assert result.patterns[0].pattern_type == "duplicate_observation"


def test_related_candidates_are_grouped_by_normalized_topic():
    loneliness = candidate(
        "cm-1",
        content="Student experiences college as a lonely environment.",
        subtopic="loneliness",
    )
    support = candidate(
        "cm-2",
        content="Student lacks a close person they feel comfortable calling.",
        subtopic="social_support",
    )

    result = DataAgent().consolidate([loneliness, support])

    assert len(result.promoted_memories) == 1
    assert result.promoted_memories[0].topic == "social_connection"
    assert result.promoted_memories[0].support_count == 2
    assert result.patterns[0].pattern_type == "recurring_theme"


def test_weak_candidate_is_rejected():
    weak = candidate("cm-weak", importance=0.25, confidence=0.40)

    result = DataAgent().consolidate([weak])

    assert result.promoted_memories == []
    assert result.rejected_memory_ids == ["cm-weak"]


def test_emotional_information_is_preserved_and_merged():
    first = candidate(
        "cm-1",
        emotions=[CandidateMemoryEmotion(emotion="loneliness", confidence=0.91)],
    )
    second = candidate(
        "cm-2",
        subtopic="social_support",
        emotions=[
            CandidateMemoryEmotion(emotion="loneliness", confidence=0.81),
            CandidateMemoryEmotion(emotion="sadness", confidence=0.72),
        ],
    )

    result = DataAgent().consolidate([first, second])
    emotions = {item.emotion: item.confidence for item in result.promoted_memories[0].emotional_state}

    assert set(emotions) == {"loneliness", "sadness"}
    assert emotions["loneliness"] >= 0.81
    assert {item.emotion for item in result.patterns[0].emotional_state} == set(emotions)


def test_evidence_ids_are_preserved_across_outputs():
    result = DataAgent().consolidate([
        candidate("cm-2", subtopic="social_support"),
        candidate("cm-1", subtopic="loneliness"),
    ])

    assert result.promoted_memories[0].evidence_ids == ["cm-1", "cm-2"]
    assert result.patterns[0].evidence_ids == ["cm-1", "cm-2"]
    assert result.relationships[0].evidence_ids == ["cm-1", "cm-2"]


def test_user_isolation_prevents_cross_user_merges():
    same_observation = "Student often feels lonely at college."
    result = DataAgent().consolidate([
        candidate("user-a-memory", user_id="user-a", content=same_observation),
        candidate("user-b-memory", user_id="user-b", content=same_observation),
    ])

    assert len(result.promoted_memories) == 2
    assert {memory.user_id for memory in result.promoted_memories} == {"user-a", "user-b"}
    assert all(memory.support_count == 1 for memory in result.promoted_memories)
    assert result.patterns == []


def test_missing_user_id_is_rejected_before_comparison():
    unscoped = candidate("cm-unscoped")
    unscoped.user_id = None

    with pytest.raises(ValueError, match="must have user_id"):
        DataAgent().consolidate([unscoped])
