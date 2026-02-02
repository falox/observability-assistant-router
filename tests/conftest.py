"""Pytest configuration and fixtures."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest


@pytest.fixture(autouse=True)
def test_config_path(monkeypatch):
    """Create a temporary config file and set ROUTER_CONFIG_PATH for all tests."""
    yaml_content = """
default_agent:
  id: "default-agent"

session:
  sticky_enabled: true
  timeout_minutes: 30
  topic_drift_threshold: 0.5

agents:
  - id: "default-agent"
    name: "Default Agent"
    handles:
      - "assistant"
    url: "http://test-default-agent:8080"
    protocol: "a2a"
    routing:
      priority: 100
      threshold: 0.3
      examples:
        - "general question"
        - "help me"
    description: "Default agent for testing"

  - id: "test-agent"
    name: "Test Agent"
    handles:
      - "test"
      - "debug"
    url: "http://test-agent:8080"
    protocol: "a2a"
    routing:
      priority: 1
      threshold: 0.8
      examples:
        - "test query"
    description: "A test agent for testing"
"""
    with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config_path = f.name

    monkeypatch.setenv("ROUTER_CONFIG_PATH", config_path)

    # Clear the lru_cache for settings and agents_config
    from router.config.agents import get_agents_config
    from router.config.settings import get_settings

    get_settings.cache_clear()
    get_agents_config.cache_clear()

    yield config_path

    # Cleanup
    Path(config_path).unlink(missing_ok=True)
    get_settings.cache_clear()
    get_agents_config.cache_clear()
