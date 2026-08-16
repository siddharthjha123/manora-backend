"""
MANORA Tests - End-to-End Interaction Flow Integration.
Validates the complete lifecycle:
User -> POST /interactions -> Interaction Service -> Memory Engine -> Emotion Agent ->
Mock Data Agent -> State Engine -> Buddy Agent -> Response.
"""

import pytest
from fastapi.testclient import TestClient
from main import app
from services.interaction_service import interaction_service


class TestInteractionFlowIntegration:
    """Integration test suite for the complete conversational interaction lifecycle."""

    @pytest.mark.asyncio
    async def test_interaction_service_full_pipeline(self):
        """Tests InteractionService orchestration directly."""
        user_id = "test-user-001"
        session_id = "test-session-001"
        text = "I planned to study at 10 but I ended up watching a series for two hours. I know I should study but this keeps happening."

        response = await interaction_service.process_interaction(
            user_id=user_id,
            session_id=session_id,
            text=text,
        )

        assert "interaction_id" in response
        assert "emotion" in response
        assert "buddy_state" in response
        assert "buddy" in response

        # Validate emotion output
        emotion = response["emotion"]
        assert "primary_emotion" in emotion
        assert "emotions" in emotion
        assert "behavioral_signals" in emotion
        assert "goal_relevance" in emotion

        # Validate buddy state
        state = response["buddy_state"]
        for key in ["happiness", "sadness", "frustration", "concern", "warmth", "patience", "energy"]:
            assert key in state
            assert 0.0 <= state[key] <= 1.0

        # Validate buddy response
        buddy = response["buddy"]
        assert "text" in buddy
        assert "expression" in buddy
        assert "intensity" in buddy
        assert "response_type" in buddy
        assert len(buddy["text"]) > 0

    def test_post_interactions_api_endpoint(self):
        """Tests POST /interactions via FastAPI TestClient."""
        client = TestClient(app)
        payload = {
            "user_id": "test-api-user-001",
            "session_id": "test-api-session-001",
            "text": "I planned to study at 10 but I ended up watching a series for two hours. I know I should study but this keeps happening.",
        }

        response = client.post("/interactions", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert "interaction_id" in data
        assert "emotion" in data
        assert "buddy_state" in data
        assert "buddy" in data

    def test_post_emotions_analyze_endpoint(self):
        """Tests POST /emotions/analyze development endpoint."""
        client = TestClient(app)
        response = client.post("/emotions/analyze", json={"text": "I am really stressed about placements."})
        assert response.status_code == 200
        data = response.json()
        assert "primary_emotion" in data
        assert "emotional_summary" in data

    def test_get_buddy_state_endpoints(self):
        """Tests GET /buddy/state/{user_id} and GET /buddy/state/{user_id}/history."""
        client = TestClient(app)
        user_id = "test-state-user-999"

        # 1. Fetch initial state
        res_initial = client.get(f"/buddy/state/{user_id}")
        assert res_initial.status_code == 200
        data_initial = res_initial.json()
        assert "state" in data_initial

        # 2. Trigger an interaction to update state
        client.post(
            "/interactions",
            json={"user_id": user_id, "session_id": "sess-1", "text": "I keep avoiding my study tasks."},
        )

        # 3. Check state updated
        res_updated = client.get(f"/buddy/state/{user_id}")
        assert res_updated.status_code == 200
        data_updated = res_updated.json()
        assert data_updated["state"]["concern"] >= data_initial["state"]["concern"]

        # 4. Fetch history
        res_history = client.get(f"/buddy/state/{user_id}/history")
        assert res_history.status_code == 200
        data_history = res_history.json()
        assert isinstance(data_history, list)
        assert len(data_history) >= 1

    def test_get_health_endpoint(self):
        """Tests GET /health endpoint."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
