"""
MANORA Backend - Sentry Exception & Error Monitoring Integration.
Provides centralized error monitoring, exception capturing, and environment tagging.
"""

import logging
from typing import Optional

from config.settings import get_settings

logger = logging.getLogger("manora.observability.sentry")

_sentry_initialized: bool = False


def init_sentry(app: Optional[object] = None) -> bool:
    """
    Initializes Sentry SDK if SENTRY_DSN is configured.
    
    Requirements:
    - Reads DSN from application settings.
    - Idempotent: safe to call multiple times without re-initializing.
    - Gracefully skips initialization if DSN is absent.
    - Disables automatic PII capture (send_default_pii=False).
    - Masks/hides credentials in logs.
    """
    global _sentry_initialized

    if _sentry_initialized:
        logger.debug("Sentry already initialized; skipping.")
        return True

    settings = get_settings()
    dsn = (settings.SENTRY_DSN or "").strip()

    if not dsn:
        logger.info("Sentry DSN not configured. Sentry monitoring is disabled.")
        return False

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        # Configure logging integration to capture ERROR level as Sentry events
        logging_integration = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR,
        )

        sentry_sdk.init(
            dsn=dsn,
            environment=settings.ENVIRONMENT,
            release=f"{settings.APP_NAME}@{settings.APP_VERSION}",
            traces_sample_rate=getattr(settings, "SENTRY_TRACES_SAMPLE_RATE", 1.0),
            send_default_pii=False,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                StarletteIntegration(transaction_style="endpoint"),
                logging_integration,
            ],
            # Filter out sensitive authorization / cookies / student content before sending
            before_send=_filter_sensitive_data,
        )

        _sentry_initialized = True
        logger.info("Sentry monitoring successfully initialized for environment '%s'.", settings.ENVIRONMENT)
        return True

    except Exception as exc:
        logger.error("Failed to initialize Sentry: %s", exc)
        return False


def _filter_sensitive_data(event: dict, hint: dict) -> Optional[dict]:
    """Sanitizes request headers, tokens, and payloads before sending to Sentry."""
    if not event:
        return event

    request_info = event.get("request", {})
    if request_info:
        headers = request_info.get("headers", {})
        for header_key in list(headers.keys()):
            if header_key.lower() in ("authorization", "cookie", "x-api-key", "set-cookie"):
                headers[header_key] = "[FILTERED]"

        # Redact potentially sensitive query params / body fields if present
        if "data" in request_info and isinstance(request_info["data"], dict):
            for k in list(request_info["data"].keys()):
                if any(secret_term in k.lower() for secret_term in ("token", "secret", "password", "key", "auth")):
                    request_info["data"][k] = "[FILTERED]"

    return event


def capture_exception(exc: Exception, **kwargs) -> Optional[str]:
    """Helper to capture an exception in Sentry if enabled."""
    if not _sentry_initialized:
        return None
    try:
        import sentry_sdk
        return sentry_sdk.capture_exception(exc, **kwargs)
    except Exception as e:
        logger.warning("Error reporting exception to Sentry: %s", e)
        return None


def capture_message(message: str, level: str = "info", **kwargs) -> Optional[str]:
    """Helper to capture a message in Sentry if enabled."""
    if not _sentry_initialized:
        return None
    try:
        import sentry_sdk
        return sentry_sdk.capture_message(message, level=level, **kwargs)
    except Exception as e:
        logger.warning("Error reporting message to Sentry: %s", e)
        return None
