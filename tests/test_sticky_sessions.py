"""Integration tests for sticky sessions in the AG-UI endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient

from router.config import (
    AgentConfig,
    AgentProtocol,
    AgentRoutingConfig,
    AgentsConfig,
    SessionConfig,
)
from router.main import app
from router.routing import SemanticRouter
from router.session import SessionStore


def make_message(content: str, msg_id: str = "msg-1") -> dict:
    """Create a message dict with required fields."""
    return {"id": msg_id, "role": "user", "content": content}


# Use module scope to avoid reloading the model for each test
@pytest.fixture(scope="module")
def shared_agents_config():
    """Create test agents configuration."""
    return AgentsConfig(
        session=SessionConfig(
            sticky_enabled=True,
            timeout_minutes=30,
            topic_drift_threshold=0.5,
        ),
        default_agent={"id": "default"},
        agents=[
            AgentConfig(
                id="default",
                name="Default Agent",
                handles=["assistant"],
                url="http://default.local",
                protocol=AgentProtocol.A2A,
                routing=AgentRoutingConfig(
                    priority=100,
                    threshold=0.3,
                    examples=["general question", "help me"],
                ),
            ),
            AgentConfig(
                id="troubleshooting",
                name="Troubleshooting Agent",
                handles=["troubleshoot", "debug", "rca"],
                url="http://troubleshooting.local",
                protocol=AgentProtocol.A2A,
                routing=AgentRoutingConfig(
                    priority=1,
                    threshold=0.7,
                    examples=[
                        "Why is my pod crashing?",
                        "Debug the CrashLoopBackOff error",
                        "Help me fix this deployment failure",
                    ],
                ),
            ),
            AgentConfig(
                id="metrics",
                name="Metrics Agent",
                handles=["metrics", "prometheus"],
                url="http://metrics.local",
                protocol=AgentProtocol.AG_UI,
                routing=AgentRoutingConfig(
                    priority=2,
                    threshold=0.7,
                    examples=[
                        "Show me CPU usage",
                        "Query Prometheus metrics",
                        "Display memory consumption",
                    ],
                ),
            ),
        ],
    )


@pytest.fixture(scope="module")
def shared_semantic_router(shared_agents_config):
    """Create test semantic router (module-scoped to avoid reloading model)."""
    router = SemanticRouter()
    router.load_model()
    router.build_index(shared_agents_config)
    return router


class MockAgentProxy:
    """Mock agent proxy that yields test events."""

    async def forward_request(self, agent, request, headers=None):
        from router.agui.models import EventType, RunFinishedEvent, RunStartedEvent

        yield RunStartedEvent(
            type=EventType.RUN_STARTED, thread_id=request.thread_id, run_id="run-1"
        )
        yield RunFinishedEvent(
            type=EventType.RUN_FINISHED, thread_id=request.thread_id, run_id="run-1"
        )

    async def close(self):
        pass


@pytest.fixture
def session_store():
    """Create test session store (fresh for each test)."""
    return SessionStore(timeout_minutes=30)


@pytest.fixture
def configured_app(shared_agents_config, shared_semantic_router, session_store):
    """Configure app state for testing and return session store."""
    mock_proxy = MockAgentProxy()

    # Save original state
    orig_agents = getattr(app.state, "agents_config", None)
    orig_loaded = getattr(app.state, "config_loaded", None)
    orig_router = getattr(app.state, "semantic_router", None)
    orig_ready = getattr(app.state, "router_ready", None)
    orig_store = getattr(app.state, "session_store", None)
    orig_proxy = getattr(app.state, "agent_proxy", None)

    # Set test state
    app.state.agents_config = shared_agents_config
    app.state.config_loaded = True
    app.state.semantic_router = shared_semantic_router
    app.state.router_ready = True
    app.state.session_store = session_store
    app.state.agent_proxy = mock_proxy

    yield session_store

    # Restore original state
    app.state.agents_config = orig_agents
    app.state.config_loaded = orig_loaded
    app.state.semantic_router = orig_router
    app.state.router_ready = orig_ready
    app.state.session_store = orig_store
    app.state.agent_proxy = orig_proxy


class TestStickySessionRouting:
    """Tests for sticky session routing behavior."""

    @pytest.fixture
    async def client(self, configured_app):
        """Create an async test client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_request_creates_session(self, configured_app, client):
        """Test that a request creates a session."""
        session_store = configured_app

        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "thread-123",
                "messages": [make_message("Why is my pod crashing?")],
            },
        )

        assert response.status_code == 200

        # Check session was created
        session = session_store.get("thread-123")
        assert session is not None
        assert session.agent_id == "troubleshooting"

    @pytest.mark.asyncio
    async def test_mention_routes_to_specific_agent(self, configured_app, client):
        """Test that @mention routes to the specified agent."""
        session_store = configured_app

        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "thread-mention",
                "messages": [make_message("@metrics why is my pod crashing?")],
            },
        )

        assert response.status_code == 200

        # Check session was created for metrics agent (not troubleshooting)
        session = session_store.get("thread-mention")
        assert session is not None
        assert session.agent_id == "metrics"

    @pytest.mark.asyncio
    async def test_mention_case_insensitive(self, configured_app, client):
        """Test that @mentions are case-insensitive."""
        session_store = configured_app

        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "thread-case",
                "messages": [make_message("@METRICS show CPU")],
            },
        )

        assert response.status_code == 200

        session = session_store.get("thread-case")
        assert session is not None
        assert session.agent_id == "metrics"

    @pytest.mark.asyncio
    async def test_default_agent_mention(self, configured_app, client):
        """Test that @assistant routes to default agent."""
        session_store = configured_app

        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "thread-default",
                "messages": [make_message("@assistant help me")],
            },
        )

        assert response.status_code == 200

        session = session_store.get("thread-default")
        assert session is not None
        # The default agent has id="default" in our test config
        assert session.agent_id == "default"

    @pytest.mark.asyncio
    async def test_unknown_mention_uses_semantic_routing(self, configured_app, client):
        """Test that unknown @mentions fall back to semantic routing."""
        session_store = configured_app

        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "thread-unknown",
                "messages": [make_message("@unknown Why is my pod crashing?")],
            },
        )

        assert response.status_code == 200

        # Should fall back to troubleshooting via semantic match
        session = session_store.get("thread-unknown")
        assert session is not None
        assert session.agent_id == "troubleshooting"
