"""Test health and root endpoints."""

from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from router.main import app


@asynccontextmanager
async def lifespan_wrapper(app):
    """Wrap app lifespan for testing."""
    async with app.router.lifespan_context(app):
        yield


@pytest.fixture
async def client():
    """Create async test client with lifespan."""
    async with lifespan_wrapper(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


async def test_root(client: AsyncClient):
    """Test root endpoint returns API metadata."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "observability-assistant-router"
    assert "version" in data


async def test_liveness(client: AsyncClient):
    """Test liveness probe returns ok."""
    response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readiness(client: AsyncClient):
    """Test readiness probe returns ok."""
    response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
