"""
MANORA Backend - Langfuse LLM & AI Observability Integration.

Langfuse v4-compatible instrumentation for:
- LLM generations
- Prompt traces
- Latency
- Model usage
- Multi-agent pipeline observations
"""

import datetime
import logging
from typing import Any, Dict, List, Optional

from config.settings import get_settings

logger = logging.getLogger("manora.observability.langfuse")

_langfuse_client: Optional[Any] = None
_langfuse_initialized: bool = False


class NoOpSpan:
    """No-op fallback observation when Langfuse is disabled or unavailable."""

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

    def start_as_current_observation(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class NoOpTrace(NoOpSpan):
    """No-op fallback root observation."""

    pass


def init_langfuse() -> Optional[Any]:
    """
    Initialize the Langfuse v4 SDK client.

    Initialization is lazy and idempotent.
    Missing credentials disable Langfuse gracefully.
    """

    global _langfuse_client, _langfuse_initialized

    if _langfuse_initialized:
        return _langfuse_client

    settings = get_settings()

    public_key = (settings.LANGFUSE_PUBLIC_KEY or "").strip()
    secret_key = (settings.LANGFUSE_SECRET_KEY or "").strip()
    host = (
        settings.LANGFUSE_BASE_URL or "https://cloud.langfuse.com"
    ).strip()

    if not public_key or not secret_key:
        logger.info(
            "Langfuse credentials not configured. "
            "Langfuse tracing is disabled."
        )

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

        logger.info(
            "Langfuse AI observability initialized successfully "
            "(host=%s).",
            host,
        )

        return _langfuse_client

    except Exception as exc:
        logger.error(
            "Failed to initialize Langfuse client: %s",
            exc,
        )

        _langfuse_initialized = True
        _langfuse_client = None

        return None


def get_langfuse_client() -> Optional[Any]:
    """Return the initialized Langfuse client."""

    global _langfuse_client

    if not _langfuse_initialized:
        return init_langfuse()

    return _langfuse_client


def is_langfuse_enabled() -> bool:
    """Return True when Langfuse is configured and initialized."""

    return get_langfuse_client() is not None


class _SafeTrace:
    """
    Thin, safe wrapper around a real Langfuse trace / span object.

    Every public method is guarded by try/except so an unexpected SDK
    shape change can never propagate into application request handling.
    """

    def __init__(self, real_obj: Any) -> None:
        self._obj = real_obj

    def update(self, **kwargs) -> "  _SafeTrace":
        try:
            if hasattr(self._obj, "update"):
                self._obj.update(**kwargs)
        except Exception as exc:
            logger.debug("Langfuse trace.update() ignored: %s", exc)
        return self

    def end(self, **kwargs) -> "_SafeTrace":
        try:
            if hasattr(self._obj, "end"):
                self._obj.end(**kwargs)
        except Exception as exc:
            logger.debug("Langfuse trace.end() ignored: %s", exc)
        return self

    def span(self, **kwargs) -> "_SafeTrace":
        try:
            if hasattr(self._obj, "span"):
                return _SafeTrace(self._obj.span(**kwargs))
        except Exception as exc:
            logger.debug("Langfuse trace.span() ignored: %s", exc)
        return _SafeTrace(NoOpSpan())

    def generation(self, **kwargs) -> "_SafeTrace":
        try:
            if hasattr(self._obj, "generation"):
                return _SafeTrace(self._obj.generation(**kwargs))
        except Exception as exc:
            logger.debug("Langfuse trace.generation() ignored: %s", exc)
        return _SafeTrace(NoOpSpan())


def create_trace(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    input_data: Optional[Any] = None,
) -> Any:
    """
    Create a root Langfuse trace and return a safe handle.

    Uses ``client.trace()`` which returns a ``StatefulTraceClient``
    **directly** (not a context manager), so the returned object is
    always safe to call ``.update()`` on without entering it.

    Falls back to NoOpTrace when Langfuse is disabled or on any error.
    """
    client = get_langfuse_client()

    if not client:
        return NoOpTrace()

    try:
        # client.trace() returns a StatefulTraceClient directly —
        # it is NOT a context manager and does NOT need `with`.
        trace_obj = client.trace(
            name=name,
            user_id=str(user_id) if user_id else None,
            session_id=str(session_id) if session_id else None,
            tags=tags or [],
            metadata=metadata or {},
            input=input_data,
        )
        return _SafeTrace(trace_obj)

    except Exception as exc:
        logger.warning(
            "Error creating Langfuse trace '%s': %s",
            name,
            exc,
        )
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
    Record an LLM generation using the Langfuse v4 observation API.

    If trace_instance is supplied, the generation becomes a child
    observation of the active trace.
    """

    client = get_langfuse_client()

    if not client:
        return NoOpSpan()

    try:
        with client.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=messages,
            metadata=metadata or {},
        ) as generation:

            if output is not None:
                generation.update(
                    output=output,
                )

            if usage:
                generation.update(
                    usage=usage,
                )

            return generation

    except Exception as exc:
        logger.warning(
            "Error recording Langfuse generation '%s': %s",
            name,
            exc,
        )

        return NoOpSpan()


def flush_langfuse() -> None:
    """Flush queued Langfuse events."""

    client = get_langfuse_client()

    if not client:
        return

    try:
        client.flush()

    except Exception as exc:
        logger.warning(
            "Error flushing Langfuse client: %s",
            exc,
        )