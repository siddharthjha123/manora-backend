"""
MANORA Backend - Prometheus Metrics Definitions.
Defines application, LLM, emotion, memory, and interaction performance counters and histograms.
"""

import logging
from prometheus_client import Counter, Histogram, REGISTRY

logger = logging.getLogger("manora.observability.metrics")


def _get_or_create_counter(name: str, documentation: str, labelnames=()):
    """Safely retrieves existing Counter or registers a new one to prevent duplicate registration errors."""
    try:
        return Counter(name, documentation, labelnames=labelnames)
    except ValueError:
        # Collector already registered in REGISTRY
        collector = REGISTRY._names_to_collectors.get(name)
        if collector:
            return collector
        # Fallback search
        for c in REGISTRY._collector_to_names:
            if name in REGISTRY._collector_to_names[c]:
                return c
        raise


def _get_or_create_histogram(name: str, documentation: str, labelnames=(), buckets=Histogram.DEFAULT_BUCKETS):
    """Safely retrieves existing Histogram or registers a new one to prevent duplicate registration errors."""
    try:
        return Histogram(name, documentation, labelnames=labelnames, buckets=buckets)
    except ValueError:
        collector = REGISTRY._names_to_collectors.get(name)
        if collector:
            return collector
        for c in REGISTRY._collector_to_names:
            if name in REGISTRY._collector_to_names[c]:
                return c
        raise


# ==============================================================================
# MANORA Application Metrics
# ==============================================================================

# Interactions
INTERACTIONS_TOTAL = _get_or_create_counter(
    "manora_interactions_total",
    "Total count of processed student interactions",
    labelnames=["status"],
)

# Emotion Predictions
EMOTION_PREDICTIONS_TOTAL = _get_or_create_counter(
    "manora_emotion_predictions_total",
    "Total emotion predictions performed",
    labelnames=["primary_emotion"],
)

# Memory Retrievals
MEMORY_RETRIEVALS_TOTAL = _get_or_create_counter(
    "manora_memory_retrievals_total",
    "Total memory retrieval queries",
    labelnames=["status", "retrieval_performed"],
)

MEMORY_RETRIEVAL_LATENCY_SECONDS = _get_or_create_histogram(
    "manora_memory_retrieval_latency_seconds",
    "Memory retrieval latency in seconds",
    labelnames=["operation"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# LLM Metrics
LLM_REQUESTS_TOTAL = _get_or_create_counter(
    "manora_llm_requests_total",
    "Total LLM generation requests",
    labelnames=["model", "status"],
)

LLM_ERRORS_TOTAL = _get_or_create_counter(
    "manora_llm_errors_total",
    "Total LLM generation errors",
    labelnames=["model", "error_type"],
)

LLM_LATENCY_SECONDS = _get_or_create_histogram(
    "manora_llm_latency_seconds",
    "LLM request latency in seconds",
    labelnames=["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

# Buddy State Updates
BUDDY_STATE_UPDATES_TOTAL = _get_or_create_counter(
    "manora_buddy_state_updates_total",
    "Total buddy state updates",
    labelnames=["status"],
)
