"""Deterministic integration tests for /interactions memory orchestration."""

import uuid

import pytest
from fastapi.testclient import TestClient

import api.interactions as interactions_api
from buddy.buddy_schema import BuddyResponse
from data_agent.data_engine import DataAgent
from data_agent.data_schema import (
    CandidateMemory,
    ConsolidatedMemory,
    MemoryDecision,
    MemoryDecisionType,
    MemoryPipelineResult,
    PersistenceResult,
    Stage1ConsolidationResult,
)
from emotion_agent.emotion_schema import EmotionAnalysis, EmotionItem
from main import app
from memory.memory_engine import MemoryEngine
from services.interaction_service import InteractionService
from state.state_engine import StateEngine


USER_ID = "550e8400-e29b-41d4-a716-446655440000"
SESSION_ID = "660e8400-e29b-41d4-a716-446655440000"
CONVERSATIONAL_MEMORY = {
    "memory_id": "conversation-memory-001",
    "text": "Context retrieved specifically for the current conversation.",
}


def emotion_analysis(interaction_id="fixture-interaction"):
    return EmotionAnalysis(
        interaction_id=interaction_id,
        primary_emotion="loneliness",
        emotions=[
            {
                "emotion": "loneliness",
                "intensity": 0.82,
                "confidence": 0.91,
                "source": "user_stated",
            }
        ],
        emotional_summary="The student feels socially isolated.",
        behavioral_signals=["withdraws when feeling isolated"],
        decision_signals=["wants to seek trusted support"],
        goal_relevance={"related": False, "goal": None},
    )


def candidate():
    return CandidateMemory(
        id="interaction-cm-001",
        user_id=USER_ID,
        content="Student feels isolated and wants trusted support.",
        emotional_state=[{"emotion": "loneliness", "confidence": 0.91}],
        importance=0.84,
        confidence=0.90,
    )


def pipeline_result(action, *, memory_id=None):
    consolidated = ConsolidatedMemory(
        consolidation_id="interaction-group-001",
        candidate_ids=["interaction-cm-001"],
        evidence_ids=["interaction-cm-001"],
        content="Student feels isolated and wants trusted support.",
        emotions=[{"emotion": "loneliness", "confidence": 0.91}],
        importance=0.84,
        confidence=0.90,
    )
    persistent = action in {MemoryDecisionType.CREATE, MemoryDecisionType.UPDATE}
    final_id = memory_id or (
        "770e8400-e29b-41d4-a716-446655440000" if persistent else None
    )
    decision = MemoryDecision(
        action=action,
        memory_id=memory_id if action == MemoryDecisionType.UPDATE else None,
        candidate_ids=consolidated.candidate_ids,
        evidence_ids=consolidated.evidence_ids,
        content=consolidated.content if persistent else None,
        emotions=consolidated.emotions,
        importance=consolidated.importance,
        confidence=consolidated.confidence,
        reasoning=f"Deterministic {action.value} fixture.",
    )
    return MemoryPipelineResult(
        stage1=Stage1ConsolidationResult(
            user_id=USER_ID,
            consolidated_memories=[consolidated],
            reasoning_summary="Deterministic orchestration fixture.",
        ),
        decisions=[decision],
        persistence=[
            PersistenceResult(
                action=action,
                memory_id=final_id,
                postgres_operation=(
                    action.value if persistent else "NONE"
                ),
                qdrant_operation="UPSERT" if persistent else "NONE",
            )
        ],
        consolidation_memory_map=(
            {consolidated.consolidation_id: final_id} if persistent else {}
        ),
    )


class FakeDatabase:
    def __init__(self, events):
        self.events = events
        self.interactions = []
        self.analyses = []
        self.state_history = []

    async def save_interaction(self, **values):
        self.events.append(f"save_{values['role']}")
        record = {
            "id": values.get("interaction_id", str(uuid.uuid4())),
            "user_id": values["user_id"],
            "session_id": values["session_id"],
            "role": values["role"],
            "raw_text": values["raw_text"],
        }
        self.interactions.append(record)
        return record

    async def get_recent_interactions(self, **values):
        self.events.append("recent_context")
        return [{"role": "user", "raw_text": "Earlier conversation"}]

    async def get_user_goals(self, **values):
        self.events.append("goals")
        return [{"title": "build supportive friendships"}]

    async def save_interaction_analysis(self, **values):
        self.events.append("save_analysis")
        self.analyses.append(values)

    async def get_buddy_state(self, **values):
        return None

    async def upsert_buddy_state(self, **values):
        self.events.append("save_state")
        return values["state_dict"]

    async def save_buddy_state_history(self, **values):
        self.events.append("save_state_history")
        self.state_history.append(values)


class FakeClassifier:
    def predict(self, text):
        return {"sadness": 0.8}


class FakeEmotionAgent:
    def __init__(self, events):
        self.events = events
        self.last_analysis = None
        self.memories = None

    async def analyze(self, **values):
        self.events.append("emotion")
        self.memories = values["memories"]
        self.last_analysis = emotion_analysis(values["interaction_id"])
        return self.last_analysis


class FakeMemoryEngine:
    def __init__(self, events, candidates, result, *, retrieve=True):
        self.events = events
        self.candidates = candidates
        self.result = result
        self.retrieve = retrieve
        self.extract_analysis = None
        self.process_calls = []
        self.retrieve_calls = []

    def should_retrieve(self, **values):
        return self.retrieve

    async def retrieve_context(self, **values):
        self.events.append("conversation_retrieval")
        self.retrieve_calls.append(values)
        return {
            "memories": [CONVERSATIONAL_MEMORY],
            "graph_context": [],
            "retrieval_performed": True,
        }

    def empty_context(self):
        return {"memories": [], "graph_context": [], "retrieval_performed": False}

    def extract_candidate_memories(self, *, interaction, emotion_analysis):
        self.events.append("candidate_extraction")
        self.extract_analysis = emotion_analysis
        return self.candidates

    async def process_long_term_memories(self, candidate_memories):
        self.events.append("memory_pipeline")
        self.process_calls.append(candidate_memories)
        return self.result


class FakeBuddy:
    def __init__(self, events):
        self.events = events
        self.memories = None

    async def generate(self, **values):
        self.events.append("buddy")
        self.memories = values["memories"]
        return BuddyResponse(
            text="I am here with you.",
            expression="encouraging",
            intensity=0.8,
            response_type="support",
        )


class TrackingStateEngine(StateEngine):
    def __init__(self, events):
        self.events = events

    def update_state(self, current_state, emotion_analysis):
        self.events.append("state")
        return super().update_state(current_state, emotion_analysis)


def build_service(*, candidates, result=None, retrieve=True):
    events = []
    database = FakeDatabase(events)
    memory = FakeMemoryEngine(events, candidates, result, retrieve=retrieve)
    emotion = FakeEmotionAgent(events)
    buddy = FakeBuddy(events)
    service = InteractionService(
        database=database,
        classifier=FakeClassifier(),
        memory=memory,
        emotion=emotion,
        state=TrackingStateEngine(events),
        buddy=buddy,
    )
    return service, database, memory, emotion, buddy, events


def assert_response_contract(response):
    assert set(response) == {"interaction_id", "emotion", "buddy_state", "buddy"}
    assert response["buddy"]["text"] == "I am here with you."


def test_production_candidate_extraction_skips_positive_greeting():
    analysis = emotion_analysis()
    analysis.primary_emotion = "joy"
    analysis.emotions = [
        EmotionItem(
            emotion="joy",
            intensity=0.95,
            confidence=0.94,
            source="model_inferred",
        )
    ]
    analysis.behavioral_signals = [
        "Initiated contact with a friendly greeting",
        "Demonstrated positive affective state",
        "Showed willingness to engage in supportive dialogue",
    ]
    analysis.decision_signals = []

    candidates = DataAgent().process(
        {"raw_text": "Hello, how are you today?"},
        analysis,
    )

    assert candidates == []


def test_production_candidate_extraction_preserves_distress_statement():
    analysis = emotion_analysis()

    candidates = DataAgent().process(
        {"raw_text": "I feel lonely and want someone trustworthy to talk to."},
        analysis,
    )

    assert len(candidates) == 1
    assert candidates[0].content == (
        "I feel lonely and want someone trustworthy to talk to."
    )
    assert candidates[0].emotional_state[0].emotion == "loneliness"


def test_memory_engine_assigns_candidate_evidence_id_and_user_scope():
    engine = MemoryEngine(data_agent_instance=DataAgent())

    candidates = engine.extract_candidate_memories(
        interaction={
            "user_id": USER_ID,
            "raw_text": "I feel lonely and want someone trustworthy to talk to.",
        },
        emotion_analysis=emotion_analysis(),
    )

    assert len(candidates) == 1
    assert candidates[0].user_id == USER_ID
    assert uuid.UUID(candidates[0].id)


def test_post_interactions_sends_emotion_candidates_to_real_pipeline_boundary(monkeypatch):
    result = pipeline_result(MemoryDecisionType.CREATE)
    service, _, memory, emotion, _, events = build_service(
        candidates=[candidate()],
        result=result,
    )
    monkeypatch.setattr(interactions_api, "interaction_service", service)

    response = TestClient(app).post(
        "/interactions",
        json={"user_id": USER_ID, "session_id": SESSION_ID, "text": "I feel alone."},
    )

    assert response.status_code == 200
    assert_response_contract(response.json())
    assert len(memory.process_calls) == 1
    assert memory.process_calls[0][0].id == "interaction-cm-001"
    assert memory.extract_analysis is emotion.last_analysis
    assert events.index("emotion") < events.index("candidate_extraction")
    assert events.index("memory_pipeline") < events.index("state")
    assert events.index("state") < events.index("buddy")


@pytest.mark.asyncio
async def test_no_candidates_skips_memory_pipeline_and_buddy_still_responds():
    service, _, memory, _, _, _ = build_service(candidates=[], result=None)

    response = await service.process_interaction(USER_ID, SESSION_ID, "Hello there.")

    assert_response_contract(response)
    assert memory.process_calls == []


@pytest.mark.asyncio
async def test_update_decision_passes_through_interaction_pipeline():
    result = pipeline_result(MemoryDecisionType.UPDATE, memory_id="mem_001")
    service, _, memory, _, _, _ = build_service(
        candidates=[candidate()],
        result=result,
    )

    response = await service.process_interaction(USER_ID, SESSION_ID, "I still feel alone.")

    assert_response_contract(response)
    assert len(memory.process_calls) == 1
    assert memory.result.decisions[0].action == MemoryDecisionType.UPDATE
    assert memory.result.decisions[0].memory_id == "mem_001"
    assert memory.result.persistence[0].postgres_operation == "UPDATE"


@pytest.mark.asyncio
async def test_rejected_candidate_creates_no_memory_and_buddy_still_responds():
    result = pipeline_result(MemoryDecisionType.REJECT)
    service, _, memory, _, _, _ = build_service(
        candidates=[candidate()],
        result=result,
    )

    response = await service.process_interaction(USER_ID, SESSION_ID, "I am eating lunch.")

    assert_response_contract(response)
    assert memory.result.decisions[0].action == MemoryDecisionType.REJECT
    assert memory.result.persistence[0].postgres_operation == "NONE"
    assert memory.result.persistence[0].qdrant_operation == "NONE"
    assert memory.result.consolidation_memory_map == {}


@pytest.mark.asyncio
async def test_buddy_uses_conversational_retrieval_not_consolidated_output():
    result = pipeline_result(MemoryDecisionType.CREATE)
    service, _, memory, emotion, buddy, _ = build_service(
        candidates=[candidate()],
        result=result,
    )

    await service.process_interaction(USER_ID, SESSION_ID, "This keeps happening again.")

    assert len(memory.retrieve_calls) == 1
    assert emotion.memories == [CONVERSATIONAL_MEMORY]
    assert buddy.memories == [CONVERSATIONAL_MEMORY]
    assert buddy.memories[0]["text"] != result.stage1.consolidated_memories[0].content
