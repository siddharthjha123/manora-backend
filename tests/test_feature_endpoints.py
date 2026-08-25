"""Deterministic endpoint tests for Memory Tree and Alternate Timeline."""

import datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.alternate_timeline as timeline_api
import api.memory_tree as memory_tree_api
from alternate_timeline.timeline_service import AlternateTimelineService
from memory.memory_tree_service import MemoryTreeService


USER_ID = "550e8400-e29b-41d4-a716-446655440000"
OTHER_USER_ID = "650e8400-e29b-41d4-a716-446655440000"


class FakeDatabase:
    def __init__(self):
        self.read_calls = []
        self.tasks = {}
        self.memories = [
            {
                "id": "11111111-1111-4111-8111-111111111111",
                "user_id": USER_ID,
                "content": "Student worries about placement outcomes.",
                "importance": 0.91,
                "confidence": 0.92,
                "emotions": [{"emotion": "stress", "confidence": 0.94}],
            },
            {
                "id": "22222222-2222-4222-8222-222222222222",
                "user_id": USER_ID,
                "content": "Student feels lonely at college.",
                "importance": 0.87,
                "confidence": 0.90,
                "emotions": [{"emotion": "loneliness", "confidence": 0.91}],
            },
            {
                "id": "33333333-3333-4333-8333-333333333333",
                "user_id": USER_ID,
                "content": "Student feels hopeful after completing a project.",
                "importance": 0.80,
                "confidence": 0.88,
                "emotions": [{"emotion": "hope", "confidence": 0.89}],
            },
        ]

    async def get_long_term_memories(self, user_id, limit=100):
        self.read_calls.append({"user_id": user_id, "limit": limit})
        return [item for item in self.memories if item["user_id"] == user_id]

    async def create_scheduled_task(
        self, user_id, title, description, scheduled_date, start_time, end_time
    ):
        task_id = f"00000000-0000-4000-8000-{len(self.tasks) + 1:012d}"
        task = {
            "id": task_id,
            "user_id": str(user_id),
            "title": title,
            "description": description,
            "scheduled_date": scheduled_date,
            "start_time": start_time,
            "end_time": end_time,
            "status": "planned",
            "decision": None,
            "reason": "",
            "created_at": datetime.datetime.now(datetime.timezone.utc),
        }
        self.tasks[task_id] = task
        return task

    async def get_scheduled_tasks(self, user_id, scheduled_date):
        return sorted(
            (
                task for task in self.tasks.values()
                if task["user_id"] == str(user_id)
                and task["scheduled_date"] == scheduled_date
            ),
            key=lambda task: task["start_time"],
        )

    async def get_scheduled_task(self, task_id):
        return self.tasks.get(str(task_id))

    async def update_scheduled_task_decision(
        self, task_id, decision, reason, status
    ):
        task = self.tasks.get(str(task_id))
        if task is None:
            return None
        task.update(decision=decision, reason=reason, status=status)
        return task


class FakeReasoningClient:
    def __init__(self):
        self.calls = []

    async def generate_json(self, *, messages, schema, **values):
        self.calls.append({"messages": messages, "schema": schema, **values})
        if schema.__name__ == "ReflectionContent":
            return schema(
                summary="Placement uncertainty is a recurring source of anxiety.",
                contributing_factors=["Placement uncertainty"],
            )
        return schema(
            baseline={
                "description": "Skipping once may provide rest but can leave work unfinished.",
                "confidence": 0.82,
            },
            events=[
                {
                    "time": "Tonight",
                    "event": "Study session skipped",
                    "likely_effect": "Immediate rest.",
                }
            ],
            summary="The main risk is repetition, not one isolated decision.",
        )


class FakeMemoryEngine:
    def __init__(self):
        self.retrieve_calls = []

    async def retrieve_context(self, *, user_id, text):
        self.retrieve_calls.append({"user_id": user_id, "text": text})
        return {
            "memories": [
                {
                    "memory_id": "11111111-1111-4111-8111-111111111111",
                    "text": "Student sometimes postpones difficult study sessions.",
                }
            ],
            "graph_context": [],
            "retrieval_performed": True,
        }


def test_memory_tree_endpoints_are_read_only_and_have_no_api_prefix(monkeypatch):
    database = FakeDatabase()
    reasoner = FakeReasoningClient()
    service = MemoryTreeService(database=database, reasoning_client=reasoner)
    monkeypatch.setattr(memory_tree_api, "memory_tree_service", service)

    app = FastAPI()
    app.include_router(memory_tree_api.router)
    client = TestClient(app)

    tree = client.get(f"/memory-tree/{USER_ID}")
    assert tree.status_code == 200
    counts = {item["emotion"]: item["memory_count"] for item in tree.json()["nodes"]}
    assert counts == {"happy": 1, "sad": 1, "angry": 0, "anxious": 1, "calm": 0}

    anxious = client.get(f"/memory-tree/{USER_ID}/emotions/anxious")
    assert anxious.status_code == 200
    assert anxious.json()["memories"][0]["memory_id"] == (
        "11111111-1111-4111-8111-111111111111"
    )

    reflection = client.post(
        f"/memory-tree/{USER_ID}/reflect",
        json={"emotion": "anxious"},
    )
    assert reflection.status_code == 200
    assert reflection.json()["reflection"]["contributing_factors"] == [
        "Placement uncertainty"
    ]
    assert len(reasoner.calls) == 1

    # The specification deliberately exposes no duplicate /api-prefixed routes.
    assert client.get(f"/api/memory-tree/{USER_ID}").status_code == 404


def test_memory_tree_rejects_unknown_frontend_category(monkeypatch):
    service = MemoryTreeService(
        database=FakeDatabase(),
        reasoning_client=FakeReasoningClient(),
    )
    monkeypatch.setattr(memory_tree_api, "memory_tree_service", service)
    app = FastAPI()
    app.include_router(memory_tree_api.router)

    response = TestClient(app).get(f"/memory-tree/{USER_ID}/emotions/confused")

    assert response.status_code == 400
    assert "Choose one of" in response.json()["detail"]


def test_alternate_timeline_task_decision_and_prediction_endpoints(monkeypatch):
    database = FakeDatabase()
    memory = FakeMemoryEngine()
    reasoner = FakeReasoningClient()
    service = AlternateTimelineService(
        database=database,
        memory=memory,
        reasoning_client=reasoner,
    )
    monkeypatch.setattr(timeline_api, "alternate_timeline_service", service)

    app = FastAPI()
    app.include_router(timeline_api.router)
    client = TestClient(app)

    created = client.post(
        "/alternate-timeline/tasks",
        json={
            "user_id": USER_ID,
            "title": "Study ML",
            "description": "Study probability and statistics",
            "date": "2026-08-19",
            "start_time": "19:00",
            "end_time": "21:00",
        },
    )
    assert created.status_code == 201
    task_id = created.json()["task_id"]
    assert created.json()["status"] == "planned"

    listed = client.get(
        f"/alternate-timeline/tasks/{USER_ID}",
        params={"date": "2026-08-19"},
    )
    assert listed.status_code == 200
    assert [task["task_id"] for task in listed.json()["tasks"]] == [task_id]

    decided = client.post(
        f"/alternate-timeline/tasks/{task_id}/decision",
        json={"decision": "skip", "reason": "I am too tired."},
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "skipped"

    predicted = client.post(
        "/alternate-timeline/predict",
        json={"user_id": USER_ID, "task_id": task_id, "scenario": "skip"},
    )
    assert predicted.status_code == 200
    assert predicted.json()["scenario"]["reason"] == "I am too tired."
    assert predicted.json()["baseline"]["confidence"] == 0.82
    assert len(memory.retrieve_calls) == 1
    assert len(reasoner.calls) == 1

    assert client.post("/api/alternate-timeline/tasks", json={}).status_code == 404


def test_alternate_timeline_prediction_enforces_task_owner(monkeypatch):
    service = AlternateTimelineService(
        database=FakeDatabase(),
        memory=FakeMemoryEngine(),
        reasoning_client=FakeReasoningClient(),
    )
    monkeypatch.setattr(timeline_api, "alternate_timeline_service", service)
    app = FastAPI()
    app.include_router(timeline_api.router)
    client = TestClient(app)
    created = client.post(
        "/alternate-timeline/tasks",
        json={
            "user_id": USER_ID,
            "title": "Study ML",
            "date": "2026-08-19",
            "start_time": "19:00",
            "end_time": "21:00",
        },
    )

    response = client.post(
        "/alternate-timeline/predict",
        json={
            "user_id": OTHER_USER_ID,
            "task_id": created.json()["task_id"],
            "scenario": "skip",
        },
    )

    assert response.status_code == 403
