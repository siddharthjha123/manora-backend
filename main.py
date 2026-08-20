"""
MANORA Backend - Main Application Entrypoint.
Digital Mental Health and Psychological Support System for Students in Higher Education.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.buddy import router as buddy_router
from api.alternate_timeline import router as alternate_timeline_router
from api.emotions import router as emotions_router
from api.chat_history import router as chat_history_router
from api.interactions import router as interactions_router
from api.memory_tree import router as memory_tree_router
from config.settings import get_settings
from database.connection import db
from observability import init_sentry, init_prometheus, init_langfuse, flush_langfuse, ObservabilityMiddleware

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("manora.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown hooks."""
    settings = get_settings()
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} ({settings.ENVIRONMENT})")
    
    # Initialize Langfuse observability on startup
    init_langfuse()
    
    await db.initialize()
    yield
    logger.info("Shutting down MANORA Backend.")
    await db.close()
    
    # Flush pending Langfuse events on shutdown
    flush_langfuse()


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Backend API for MANORA: Digital Mental Health and Psychological Support System for Students in Higher Education. "
        "Features an AI Buddy with internal emotional states, memory retrieval, ML emotion classification, and relationship tracking."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize Sentry error monitoring early
init_sentry(app)

# Add Observability and CORS middlewares
app.add_middleware(ObservabilityMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus HTTP metrics & expose /metrics endpoint
init_prometheus(app)

# Mount API routers
app.include_router(interactions_router)
app.include_router(emotions_router)
app.include_router(buddy_router)
app.include_router(memory_tree_router)
app.include_router(alternate_timeline_router)
app.include_router(chat_history_router)


@app.get("/", tags=["System"])
async def root():
    """Root endpoint welcoming clients and directing to OpenAPI documentation."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint."""
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "services": {
            "postgres": db.is_postgres_connected,
            "qdrant_enabled": settings.QDRANT_ENABLED,
            "neo4j_enabled": settings.NEO4J_ENABLED,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
