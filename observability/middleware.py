"""
MANORA Backend - Observability Middleware.
Provides correlation ID generation, request duration tracking, and safe error logging.
"""

import logging
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("manora.observability.middleware")


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware that attaches a unique X-Request-ID to each request/response,
    measures request processing time, and logs request lifecycle without sensitive data.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start_time = time.perf_counter()

        # Attach request_id to request state
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Attach headers to response
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

            # Log request summary (excluding query params and bodies containing student data)
            if not request.url.path.startswith("/metrics"):
                logger.info(
                    "HTTP %s %s -> %d (%.2fms) [req_id=%s]",
                    request.method,
                    request.url.path,
                    response.status_code,
                    duration_ms,
                    request_id,
                )

            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "HTTP %s %s -> EXCEPTION: %s (%.2fms) [req_id=%s]",
                request.method,
                request.url.path,
                exc,
                duration_ms,
                request_id,
            )
            raise
