"""
MANORA Backend - Prometheus Instrumentation Integration.
Instruments HTTP requests, response latency, and exposes the /metrics endpoint.
"""

import logging
from typing import Optional
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from config.settings import get_settings

logger = logging.getLogger("manora.observability.prometheus")

_prometheus_initialized: bool = False


def init_prometheus(app: FastAPI) -> bool:
    """
    Initializes Prometheus metrics instrumentation and mounts the /metrics endpoint.

    Requirements:
    - Exposes GET /metrics.
    - Captures request counts, status codes, and latency histograms.
    - Avoids high-cardinality labels by normalizing routes.
    - Idempotent and thread-safe.
    """
    global _prometheus_initialized

    if _prometheus_initialized:
        logger.debug("Prometheus instrumentator already initialized.")
        return True

    settings = get_settings()
    if not getattr(settings, "PROMETHEUS_ENABLED", True):
        logger.info("Prometheus metrics disabled via settings.")
        return False

    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        instrumentator = Instrumentator(
            should_group_status_codes=False,
            should_ignore_untemplated=True,
            should_round_latency_decimals=True,
            round_latency_decimals=4,
            excluded_handlers=["/metrics"],
        )

        instrumentator.instrument(app)
        instrumentator.expose(
            app=app,
            endpoint="/metrics",
            tags=["System"],
            include_in_schema=True,
        )

        _prometheus_initialized = True
        logger.info("Prometheus instrumentation and /metrics endpoint initialized.")
        return True

    except Exception as exc:
        logger.warning(
            "prometheus-fastapi-instrumentator failed (%s). Falling back to native prometheus_client endpoint.",
            exc,
        )

        # Fallback route
        @app.get("/metrics", tags=["System"], include_in_schema=True)
        async def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        _prometheus_initialized = True
        return True
