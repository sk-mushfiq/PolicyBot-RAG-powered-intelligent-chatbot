"""
app/main.py
────────────
FastAPI application entry point.
Registers all routers, adds CORS middleware, and exposes /health.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import ingest, chat
from app.api.schemas import HealthResponse
from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.vector_store import VectorStoreManager

logger = get_logger(__name__)
settings = get_settings()


def create_app() -> FastAPI:
    """Application factory — returns a configured FastAPI instance."""

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "RAG-powered chatbot for answering HR policy questions "
            "using LangChain, ChromaDB, and OpenAI."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware ─────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],    # Tighten this in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────
    app.include_router(ingest.router)
    app.include_router(chat.router)

    # ── Startup Event ──────────────────────────────
    @app.on_event("startup")
    async def on_startup():
        logger.info(
            f"🚀 {settings.app_name} v{settings.app_version} starting up..."
        )

    # ── Health Check ───────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> HealthResponse:
        """Verify the API is running and the vector store is reachable."""
        vs = VectorStoreManager()
        store_ready = not vs.collection_is_empty()
        return HealthResponse(
            status="ok",
            app_name=settings.app_name,
            version=settings.app_version,
            vector_store_ready=store_ready,
        )

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "message": f"Welcome to {settings.app_name}",
            "docs": "/docs",
        }

    return app


app = create_app()
