"""
MANORA Backend - Observability Package.
Unifies Sentry, Langfuse, Prometheus metrics, and request middleware.
"""

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
from observability.prometheus import init_prometheus
from observability.middleware import ObservabilityMiddleware
from observability import metrics

__all__ = [
    "init_sentry",
    "capture_exception",
    "capture_message",
    "init_langfuse",
    "get_langfuse_client",
    "is_langfuse_enabled",
    "create_trace",
    "record_generation",
    "flush_langfuse",
    "NoOpTrace",
    "NoOpSpan",
    "init_prometheus",
    "ObservabilityMiddleware",
    "metrics",
]
