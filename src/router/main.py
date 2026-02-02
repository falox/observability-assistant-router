"""FastAPI application entry point for the observability-assistant-router."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from router import __version__
from router.config import ConfigLoadError, ConfigReloader, get_settings, load_agents_config
from router.routing import SemanticRouter
from router.session import SessionStore

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.value),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events."""
    logger.info("Starting observability-assistant-router v%s", __version__)
    logger.info("Configuration path: %s", settings.config_path)

    # Load agents configuration
    try:
        agents_config = load_agents_config()
        app.state.agents_config = agents_config
        app.state.config_loaded = True
        logger.info("Loaded configuration with %d agents", len(agents_config.agents))
        for agent in agents_config.agents:
            logger.info(
                "  - %s (@%s) via %s",
                agent.name,
                ", @".join(agent.handles),
                agent.protocol.value,
            )
    except ConfigLoadError as e:
        logger.error("Failed to load configuration: %s", e)
        app.state.agents_config = None
        app.state.config_loaded = False

    # Initialize semantic router
    app.state.router_ready = False
    if app.state.config_loaded:
        try:
            semantic_router = SemanticRouter(model_name=settings.embedding_model)
            semantic_router.load_model()
            semantic_router.build_index(app.state.agents_config)
            app.state.semantic_router = semantic_router
            app.state.router_ready = True
            logger.info("Semantic router initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize semantic router: %s", e)
            app.state.semantic_router = None
    else:
        app.state.semantic_router = None

    # Initialize agent proxy (import here to avoid circular imports)
    from router.agents.proxy import AgentProxy

    app.state.agent_proxy = AgentProxy()
    logger.info("Agent proxy initialized")

    # Initialize session store for sticky sessions
    if settings.session_enabled:
        session_timeout = settings.session_timeout_min
        if app.state.config_loaded and app.state.agents_config.session:
            session_timeout = app.state.agents_config.session.timeout_minutes
        app.state.session_store = SessionStore(timeout_minutes=session_timeout)
        logger.info(
            "Session store initialized (timeout=%d minutes)",
            session_timeout,
        )
    else:
        app.state.session_store = None
        logger.info("Sticky sessions disabled")

    # Initialize config hot-reload watcher
    if settings.hot_reload_enabled and app.state.config_loaded:
        config_reloader = ConfigReloader(
            app=app,
            config_path=settings.config_path,
            debounce_seconds=settings.hot_reload_debounce_seconds,
        )
        config_reloader.start()
        app.state.config_reloader = config_reloader
    else:
        app.state.config_reloader = None
        if not settings.hot_reload_enabled:
            logger.info("Config hot-reload disabled")

    yield

    # Cleanup
    if hasattr(app.state, "config_reloader") and app.state.config_reloader:
        app.state.config_reloader.stop()
    if hasattr(app.state, "agent_proxy") and app.state.agent_proxy:
        await app.state.agent_proxy.close()
    logger.info("Shutting down observability-assistant-router")


app = FastAPI(
    title="Observability Assistant Router",
    description=(
        "Multi-agent orchestrator service that routes messages "
        "to specialized agents via semantic matching"
    ),
    version=__version__,
    lifespan=lifespan,
)

# Include the AG-UI protocol router (import here to avoid circular imports)
from router.agui.endpoint import agui_router  # noqa: E402

app.include_router(agui_router)


@app.get("/", response_class=JSONResponse)
async def root() -> dict:
    """API metadata endpoint."""
    return {
        "name": "observability-assistant-router",
        "version": __version__,
        "description": "Multi-agent orchestrator for observability assistants",
    }


@app.get("/health/live", response_class=JSONResponse)
async def liveness() -> dict:
    """Kubernetes liveness probe endpoint."""
    return {"status": "ok"}


@app.get("/health/ready", response_class=JSONResponse)
async def readiness() -> JSONResponse:
    """Kubernetes readiness probe endpoint."""
    if not getattr(app.state, "config_loaded", False):
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "configuration not loaded"},
        )
    if not getattr(app.state, "router_ready", False):
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "reason": "semantic router not initialized"},
        )
    return JSONResponse(content={"status": "ok"})


@app.post("/admin/reload-config", response_class=JSONResponse)
async def reload_config() -> JSONResponse:
    """Manually trigger configuration reload.

    This endpoint allows operators to trigger an immediate reload of the
    agent configuration without waiting for file system events.
    """
    config_reloader = getattr(app.state, "config_reloader", None)

    if config_reloader is None:
        # Hot reload not enabled, create a temporary reloader
        from router.config import ConfigReloader

        temp_reloader = ConfigReloader(
            app=app,
            config_path=settings.config_path,
        )
        success = temp_reloader.reload()
    else:
        success = config_reloader.reload()

    if success:
        agents_config = getattr(app.state, "agents_config", None)
        agent_count = len(agents_config.agents) if agents_config else 0
        reload_count = (
            config_reloader.reload_count if config_reloader else 1
        )
        return JSONResponse(
            content={
                "status": "ok",
                "message": "Configuration reloaded successfully",
                "agent_count": agent_count,
                "reload_count": reload_count,
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Failed to reload configuration. Check logs for details.",
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "router.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
