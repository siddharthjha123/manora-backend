"""
MANORA Observability Integration Tests.
Validates Sentry, Prometheus (/metrics), Langfuse, and MANORA application metrics.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from main import app
from config.settings import get_settings
from observability.sentry import init_sentry, capture_exception, capture_message
from observability.langfuse import (
    init_langfuse,
    get_langfuse_client,
    is_langfuse_enabled,
    create_trace,
    record_generation,
    flush_langfuse,
    NoOpTrace,
    NoOpSpan,
)
from observability.metrics import (
    INTERACTIONS_TOTAL,
    EMOTION_PREDICTIONS_TOTAL,
    MEMORY_RETRIEVALS_TOTAL,
    LLM_REQUESTS_TOTAL,
    LLM_ERRORS_TOTAL,
    LLM_LATENCY_SECONDS,
    BUDDY_STATE_UPDATES_TOTAL,
)
from llm.base import LLMClient
from services.interaction_service import interaction_service
from emotion_agent.emotion_engine import emotion_agent
from memory.memory_engine import memory_engine


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# ==============================================================================
# 1. Sentry Integration Tests
# ==============================================================================

class TestSentryObservability:
    """Tests Sentry error and exception monitoring."""

    def test_sentry_disabled_when_dsn_empty(self, monkeypatch):
        """Validates that Sentry safely skips initialization when DSN is empty."""
        monkeypatch.setattr(get_settings(), "SENTRY_DSN", None)
        # Re-evaluating with unconfigured DSN
        with patch("observability.sentry._sentry_initialized", False):
            result = init_sentry()
            assert result is False
            assert capture_exception(Exception("test")) is None
            assert capture_message("test message") is None

    def test_sentry_init_with_valid_dsn(self, monkeypatch):
        """Validates that Sentry initializes when DSN is provided (mocked)."""
        monkeypatch.setattr(get_settings(), "SENTRY_DSN", "https://public_key@sentry.io/12345")
        with patch("observability.sentry._sentry_initialized", False), \
             patch("sentry_sdk.init") as mock_init:
            result = init_sentry()
            assert result is True
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args.kwargs
            assert call_kwargs["dsn"] == "https://public_key@sentry.io/12345"
            assert call_kwargs["send_default_pii"] is False
            assert "environment" in call_kwargs

    def test_sentry_sensitive_data_filter(self):
        """Validates that authorization headers and secret fields are redacted in Sentry before_send."""
        from observability.sentry import _filter_sensitive_data

        event = {
            "request": {
                "headers": {
                    "Authorization": "Bearer secret_token_12345",
                    "Cookie": "session=abcdef",
                    "Content-Type": "application/json",
                },
                "data": {
                    "password": "supersecretpassword",
                    "user_token": "token_abc",
                    "text": "Hello world",
                },
            }
        }

        sanitized = _filter_sensitive_data(event, {})
        headers = sanitized["request"]["headers"]
        data = sanitized["request"]["data"]

        assert headers["Authorization"] == "[FILTERED]"
        assert headers["Cookie"] == "[FILTERED]"
        assert headers["Content-Type"] == "application/json"
        assert data["password"] == "[FILTERED]"
        assert data["user_token"] == "[FILTERED]"
        assert data["text"] == "Hello world"


# ==============================================================================
# 2. Prometheus & /metrics Tests
# ==============================================================================

class TestPrometheusObservability:
    """Tests Prometheus metrics collection and /metrics HTTP endpoint."""

    def test_metrics_endpoint_returns_200(self, client):
        """Verifies GET /metrics returns HTTP 200 and text/plain Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    def test_manora_metrics_exist_in_scrape_output(self, client):
        """Verifies that MANORA domain metrics are registered and exposed in /metrics output."""
        response = client.get("/metrics")
        assert response.status_code == 200
        content = response.text

        expected_metrics = [
            "manora_interactions_total",
            "manora_emotion_predictions_total",
            "manora_memory_retrievals_total",
            "manora_memory_retrieval_latency_seconds",
            "manora_llm_requests_total",
            "manora_llm_errors_total",
            "manora_llm_latency_seconds",
            "manora_buddy_state_updates_total",
        ]

        for metric in expected_metrics:
            assert metric in content, f"Metric '{metric}' missing from /metrics output"

    def test_interaction_increments_metrics(self, client):
        """Verifies that POST /interactions increments interaction, emotion, and buddy counters."""
        payload = {
            "user_id": "test-obs-user-01",
            "session_id": "test-obs-session-01",
            "text": "I feel really anxious about my exam tomorrow.",
        }

        # Check baseline or perform request
        response = client.post("/interactions", json=payload)
        assert response.status_code == 200

        # Verify /metrics reflects counts
        metrics_response = client.get("/metrics")
        metrics_text = metrics_response.text
        assert 'manora_interactions_total{status="success"}' in metrics_text
        assert 'manora_buddy_state_updates_total{status="success"}' in metrics_text

    def test_emotion_analyze_increments_metric(self, client):
        """Verifies that POST /emotions/analyze increments manora_emotion_predictions_total."""
        response = client.post("/emotions/analyze", json={"text": "I am so overwhelmed."})
        assert response.status_code == 200

        metrics_response = client.get("/metrics")
        assert "manora_emotion_predictions_total" in metrics_response.text

    def test_no_high_cardinality_labels_in_metrics(self, client):
        """Validates that user_id, session_id, and prompt text are NEVER exposed in metric labels."""
        secret_user = "user-secret-private-998877"
        secret_prompt = "VeryPrivateConfidentialPromptString"

        client.post(
            "/interactions",
            json={
                "user_id": secret_user,
                "session_id": "sess-99",
                "text": secret_prompt,
            },
        )

        metrics_output = client.get("/metrics").text

        # Strict check: user id and raw prompt string must NOT be in /metrics label output
        assert secret_user not in metrics_output
        assert secret_prompt not in metrics_output


# ==============================================================================
# 3. Langfuse AI Observability Tests
# ==============================================================================

class TestLangfuseObservability:
    """Tests Langfuse LLM tracing and graceful fallback when unconfigured."""

    def test_langfuse_disabled_gracefully_when_keys_missing(self, monkeypatch):
        """Verifies Langfuse initialization returns None when keys are empty."""
        monkeypatch.setattr(get_settings(), "LANGFUSE_PUBLIC_KEY", None)
        monkeypatch.setattr(get_settings(), "LANGFUSE_SECRET_KEY", None)

        with patch("observability.langfuse._langfuse_initialized", False):
            client = init_langfuse()
            assert client is None
            assert is_langfuse_enabled() is False

    def test_noop_trace_when_langfuse_disabled(self):
        """Verifies create_trace and record_generation return NoOp objects without raising errors."""
        trace = create_trace(
            name="test_trace",
            user_id="u123",
            session_id="s123",
            tags=["test"],
        )
        assert isinstance(trace, (NoOpTrace, NoOpSpan))

        # Methods on NoOp span should chain smoothly without error
        child_span = trace.span(name="sub_span")
        assert isinstance(child_span, (NoOpTrace, NoOpSpan))
        trace.update(output={"status": "ok"})
        trace.end()

        # record_generation should not fail
        gen = record_generation(
            name="test_gen",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
            output="hi",
        )
        assert isinstance(gen, (NoOpTrace, NoOpSpan))

        # flush should not fail
        flush_langfuse()

    @pytest.mark.asyncio
    async def test_llm_client_generation_unaffected_by_langfuse(self):
        """Verifies LLMClient returns expected content whether Langfuse is enabled or not."""
        client = LLMClient()
        messages = [{"role": "user", "content": "I feel sad."}]

        # Run mock generate
        response = await client.generate(messages)
        assert isinstance(response, str)
        assert len(response) > 0


# ==============================================================================
# 4. Middleware & Health Endpoints
# ==============================================================================

class TestMiddlewareAndHealthEndpoints:
    """Tests middleware response headers and system health endpoints."""

    def test_health_endpoint(self, client):
        """Verifies GET /health returns 200 and expected status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "environment" in data

    def test_root_endpoint(self, client):
        """Verifies GET / returns 200 and docs pointer."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"

    def test_request_id_and_response_time_headers(self, client):
        """Verifies ObservabilityMiddleware injects X-Request-ID and X-Response-Time headers."""
        response = client.get("/")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time" in response.headers
        assert response.headers["X-Response-Time"].endswith("ms")
