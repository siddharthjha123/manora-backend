"""
MANORA Backend - Langfuse LLM & AI Observability Integration.
Instruments LLM generations, prompt traces, latency, model usage, and multi-agent pipeline spans.
"""

import contextlib
import datetime
import logging
import time
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger("manora.observability.langfuse")

_langfuse_client: Optional[Any] = None
_langfuse_initialized: bool = False


class NoOpSpan:
    """No-op fallback span when Langfuse is disabled or encounters an error."""

    def span(self, *args, **kwargs):
        return self

    def generation(self, *args, **kwargs):
        return self

    def event(self, *args, **kwargs):
        return self

    def update(self, *args, **kwargs):
        return self

    def end(self, *args, **kwargs):
        return self

    def score(self, *args, **kwargs):
        return self


class NoOpTrace(NoOpSpan):
    """No-op fallback trace when Langfuse is disabled."""
    pass


def init_langfuse() -> Optional[Any]:
    """
    Initializes the Langfuse SDK client if credentials are configured.
    
    Requirements:
    - Reads keys from settings.
    - Idempotent and thread-safe.
    - Gracefully skips without error if keys are missing.
    - Never logs or exposes secret keys.
    """
    global _langfuse_client, _langfuse_initialized

    if _langfuse_initialized:
        return _langfuse_client

    settings = get_settings()
    public_key = (settings.LANGFUSE_PUBLIC_KEY or "").strip()
    secret_key = (settings.LANGFUSE_SECRET_KEY or "").strip()
    host = (settings.LANGFUSE_BASE_URL or "https://cloud.langfuse.com").strip()

    if not public_key or not secret_key:
        logger.info("Langfuse credentials not configured. Langfuse tracing is disabled.")
        _langfuse_initialized = True
        _langfuse_client = None
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        _langfuse_initialized = True
        logger.info("Langfuse AI observability successfully initialized (host=%s).", host)
        return _langfuse_client

    except Exception as exc:
        logger.error("Failed to initialize Langfuse client: %s", exc)
        _langfuse_initialized = True
        _langfuse_client = None
        return None


def get_langfuse_client() -> Optional[Any]:
    """Returns the active Langfuse client if initialized, else None."""
    global _langfuse_client
    if not _langfuse_initialized:
        return init_langfuse()
    return _langfuse_client


def is_langfuse_enabled() -> bool:
    """Returns True if Langfuse client is active and configured."""
    return get_langfuse_client() is not None


def create_trace(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    input_data: Optional[Any] = None,
) -> Any:
    """
    Creates an interaction trace in Langfuse, or returns a NoOpTrace if disabled.
    """
    client = get_langfuse_client()
    if not client:
        return NoOpTrace()

    try:
        trace = client.trace(
            name=name,
            user_id=str(user_id) if user_id else None,
            session_id=str(session_id) if session_id else None,
            tags=tags or [],
            metadata=metadata or {},
            input=input_data,
        )
        return trace
    except Exception as exc:
        logger.warning("Error creating Langfuse trace '%s': %s", name, exc)
        return NoOpTrace()


def record_generation(
    name: str,
    model: str,
    messages: List[Dict[str, str]],
    output: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    usage: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trace_instance: Optional[Any] = None,
) -> Any:
    """
    Records an LLM generation event under an existing trace or as a standalone generation.
    """
    client = get_langfuse_client()
    if not client and not trace_instance:
        return NoOpSpan()

    try:
        # Convert timestamps if provided as perf_counter or floats
        start_dt = datetime.datetime.fromtimestamp(start_time, tz=datetime.timezone.utc) if (start_time and start_time > 1e9) else None
        end_dt = datetime.datetime.fromtimestamp(end_time, tz=datetime.timezone.utc) if (end_time and end_time > 1e9) else None

        gen_kwargs: Dict[str, Any] = {
            "name": name,
            "model": model,
            "input": messages,
            "output": output,
            "metadata": metadata or {},
        }
        if start_dt:
            gen_kwargs["start_time"] = start_dt
        if end_dt:
            gen_kwargs["end_time"] = end_dt
        if usage:
            gen_kwargs["usage"] = usage

        if trace_instance and hasattr(trace_instance, "generation"):
            return trace_instance.generation(**gen_kwargs)
        elif client:
            return client.generation(**gen_kwargs)
        return NoOpSpan()

    except Exception as exc:
        logger.warning("Error recording Langfuse generation '%s': %s", name, exc)
        return NoOpSpan()


def flush_langfuse():
    """Flushes queued Langfuse events."""
    client = get_langfuse_client()
    if client and hasattr(client, "flush"):
        try:
            client.flush()
        except Exception as exc:
            logger.warning("Error flushing Langfuse client: %s", exc)
