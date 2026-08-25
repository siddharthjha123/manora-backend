# MANORA Backend Observability Import Mapping

This document provides a comprehensive mapping of all observability imports across the MANORA codebase.

---

## 1. Application Entrypoint

### [`main.py`](file:///Users/bala/manora-backend/main.py)
```python
from observability import (
    init_sentry,
    init_prometheus,
    init_langfuse,
    flush_langfuse,
    ObservabilityMiddleware,
)
```
- Initializes Sentry on application startup.
- Mounts `ObservabilityMiddleware` for correlation ID (`X-Request-ID`) and duration tracking (`X-Response-Time`).
- Initializes Langfuse and Prometheus instrumentation.
- Flushes pending Langfuse events during lifespan shutdown.

---

## 2. Core LLM Layer

### [`llm/base.py`](file:///Users/bala/manora-backend/llm/base.py)
```python
from observability.metrics import (
    LLM_REQUESTS_TOTAL,
    LLM_ERRORS_TOTAL,
    LLM_LATENCY_SECONDS,
)
from observability.langfuse import record_generation
```
- Records LLM request counts, error counts, and execution latency histograms.
- Records generation traces in Langfuse (model, input messages, output completion, token usage).

### [`data_agent/data_llm_client.py`](file:///Users/bala/manora-backend/data_agent/data_llm_client.py)
```python
from observability.metrics import (
    LLM_REQUESTS_TOTAL,
    LLM_ERRORS_TOTAL,
    LLM_LATENCY_SECONDS,
)
from observability.langfuse import record_generation
```
- Instruments Data Agent Qwen LLM calls with performance metrics and Langfuse generation events.

---

## 3. Services & Agents

### [`services/interaction_service.py`](file:///Users/bala/manora-backend/services/interaction_service.py)
```python
from observability.metrics import (
    INTERACTIONS_TOTAL,
    BUDDY_STATE_UPDATES_TOTAL,
)
from observability.langfuse import create_trace
```
- Traces the complete 8-stage student interaction lifecycle in Langfuse.
- Tracks `manora_interactions_total` and `manora_buddy_state_updates_total`.

### [`emotion_agent/emotion_engine.py`](file:///Users/bala/manora-backend/emotion_agent/emotion_engine.py)
```python
from observability.metrics import EMOTION_PREDICTIONS_TOTAL
```
- Increments `manora_emotion_predictions_total` tagged by `primary_emotion`.

### [`memory/memory_engine.py`](file:///Users/bala/manora-backend/memory/memory_engine.py)
```python
from observability.metrics import (
    MEMORY_RETRIEVALS_TOTAL,
    MEMORY_RETRIEVAL_LATENCY_SECONDS,
)
```
- Measures retrieval latency and increments `manora_memory_retrievals_total` and `manora_memory_retrieval_latency_seconds`.

---

## 4. Test Suite

### [`tests/test_observability.py`](file:///Users/bala/manora-backend/tests/test_observability.py)
```python
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
```
- Comprehensive unit and integration testing of the entire observability suite.
