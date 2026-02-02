"""Test configuration loading and validation."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from router.config import (
    AgentConfig,
    AgentProtocol,
    AgentsConfig,
    ConfigLoadError,
    DefaultAgentConfig,
    SessionConfig,
    load_agents_config,
)


class TestSessionConfig:
    """Test SessionConfig model."""

    def test_defaults(self):
        """Test default values."""
        config = SessionConfig()
        assert config.sticky_enabled is True
        assert config.timeout_minutes == 30
        assert config.topic_drift_threshold == 0.5

    def test_custom_values(self):
        """Test custom values."""
        config = SessionConfig(
            sticky_enabled=False,
            timeout_minutes=60,
            topic_drift_threshold=0.7,
        )
        assert config.sticky_enabled is False
        assert config.timeout_minutes == 60
        assert config.topic_drift_threshold == 0.7

    def test_threshold_bounds(self):
        """Test threshold must be between 0 and 1."""
        with pytest.raises(ValueError):
            SessionConfig(topic_drift_threshold=1.5)
        with pytest.raises(ValueError):
            SessionConfig(topic_drift_threshold=-0.1)

    def test_timeout_positive(self):
        """Test timeout must be positive."""
        with pytest.raises(ValueError):
            SessionConfig(timeout_minutes=0)


class TestDefaultAgentConfig:
    """Test DefaultAgentConfig model."""

    def test_required_fields(self):
        """Test required id field."""
        config = DefaultAgentConfig(id="my-default-agent")
        assert config.id == "my-default-agent"

    def test_id_validation(self):
        """Test id validation."""
        # Empty id should fail
        with pytest.raises(ValueError):
            DefaultAgentConfig(id="")


class TestAgentConfig:
    """Test AgentConfig model."""

    def test_full_config(self):
        """Test full agent config."""
        config = AgentConfig(
            id="test-agent",
            name="Test Agent",
            handles=["Test", "Debug"],
            url="http://localhost:8080",
            protocol="a2a",
            routing={"priority": 1, "threshold": 0.8, "examples": ["hello"]},
            description="A test agent",
        )
        assert config.id == "test-agent"
        assert config.name == "Test Agent"
        assert config.handles == ["test", "debug"]  # Should be lowercase
        assert str(config.url).rstrip("/") == "http://localhost:8080"
        assert config.protocol == AgentProtocol.A2A
        assert config.routing.priority == 1
        assert config.routing.threshold == 0.8
        assert config.routing.examples == ["hello"]
        assert config.description == "A test agent"

    def test_handles_lowercased(self):
        """Test handles are automatically lowercased."""
        config = AgentConfig(
            id="test",
            name="Test",
            handles=["MyAgent", "DEBUG", "RcA"],
            url="http://localhost",
            routing={"examples": ["hi"]},
        )
        assert config.handles == ["myagent", "debug", "rca"]

    def test_single_handle(self):
        """Test agent with single handle."""
        config = AgentConfig(
            id="test",
            name="Test",
            handles=["agent"],
            url="http://localhost",
            routing={"examples": ["hi"]},
        )
        assert config.handles == ["agent"]

    def test_protocol_ag_ui(self):
        """Test AG-UI protocol."""
        config = AgentConfig(
            id="test",
            name="Test",
            handles=["test"],
            url="http://localhost",
            protocol="ag-ui",
            routing={"examples": ["hi"]},
        )
        assert config.protocol == AgentProtocol.AG_UI

    def test_routing_examples_optional(self):
        """Test routing examples are optional (for fallback/default agents)."""
        config = AgentConfig(
            id="test",
            name="Test",
            handles=["test"],
            url="http://localhost",
            routing={"examples": []},
        )
        assert config.routing.examples == []

    def test_routing_section_optional(self):
        """Test routing section is optional (for fallback/default agents)."""
        config = AgentConfig(
            id="test",
            name="Test",
            handles=["test"],
            url="http://localhost",
        )
        assert config.routing is None

    def test_handles_required(self):
        """Test at least one handle is required."""
        with pytest.raises(ValueError):
            AgentConfig(
                id="test",
                name="Test",
                handles=[],
                url="http://localhost",
                routing={"examples": ["hi"]},
            )


class TestAgentsConfig:
    """Test AgentsConfig model."""

    def test_minimal_config(self):
        """Test minimal valid config with default agent in agents array."""
        config = AgentsConfig(
            default_agent={"id": "my-default"},
            agents=[
                {
                    "id": "my-default",
                    "name": "Default Agent",
                    "handles": ["assistant"],
                    "url": "http://localhost:8080",
                    "routing": {"examples": ["help me"]},
                }
            ],
        )
        assert config.session.sticky_enabled is True
        assert config.default_agent.id == "my-default"
        assert len(config.agents) == 1

    def test_full_config(self):
        """Test full config with multiple agents."""
        config = AgentsConfig(
            session={"sticky_enabled": False, "timeout_minutes": 15},
            default_agent={"id": "default-agent"},
            agents=[
                {
                    "id": "default-agent",
                    "name": "Default",
                    "handles": ["assistant"],
                    "url": "http://default:8080",
                    "routing": {"examples": ["general"]},
                },
                {
                    "id": "agent-1",
                    "name": "Agent One",
                    "handles": ["one", "first"],
                    "url": "http://agent1:8080",
                    "protocol": "a2a",
                    "routing": {
                        "priority": 1,
                        "threshold": 0.85,
                        "examples": ["test query"],
                    },
                },
            ],
        )
        assert config.session.sticky_enabled is False
        assert len(config.agents) == 2
        assert config.agents[1].id == "agent-1"
        assert config.agents[1].handles == ["one", "first"]

    def test_default_agent_must_exist_in_agents(self):
        """Test that default_agent.id must reference an existing agent."""
        with pytest.raises(ValueError, match="not found in agents"):
            AgentsConfig(
                default_agent={"id": "nonexistent"},
                agents=[
                    {
                        "id": "some-agent",
                        "name": "Some Agent",
                        "handles": ["test"],
                        "url": "http://localhost",
                        "routing": {"examples": ["test"]},
                    }
                ],
            )

    def test_get_default_agent(self):
        """Test get_default_agent method."""
        config = AgentsConfig(
            default_agent={"id": "my-default"},
            agents=[
                {
                    "id": "my-default",
                    "name": "My Default Agent",
                    "handles": ["assistant"],
                    "url": "http://localhost:8080",
                    "routing": {"examples": ["help"]},
                }
            ],
        )
        default = config.get_default_agent()
        assert default.id == "my-default"
        assert default.name == "My Default Agent"

    def test_is_default_agent(self):
        """Test is_default_agent method."""
        config = AgentsConfig(
            default_agent={"id": "my-default"},
            agents=[
                {
                    "id": "my-default",
                    "name": "Default",
                    "handles": ["assistant"],
                    "url": "http://localhost",
                    "routing": {"examples": ["help"]},
                },
                {
                    "id": "other-agent",
                    "name": "Other",
                    "handles": ["other"],
                    "url": "http://localhost",
                    "routing": {"examples": ["test"]},
                },
            ],
        )
        default = config.get_default_agent()
        other = config.get_agent_by_id("other-agent")

        assert config.is_default_agent(default) is True
        assert config.is_default_agent(other) is False

    def test_get_agent_by_id(self):
        """Test get_agent_by_id method."""
        config = AgentsConfig(
            default_agent={"id": "default"},
            agents=[
                {
                    "id": "default",
                    "name": "Default",
                    "handles": ["assistant"],
                    "url": "http://localhost",
                    "routing": {"examples": ["help"]},
                },
                {
                    "id": "agent-1",
                    "name": "Agent One",
                    "handles": ["one"],
                    "url": "http://agent1",
                    "routing": {"examples": ["test"]},
                },
            ],
        )
        agent = config.get_agent_by_id("agent-1")
        assert agent is not None
        assert agent.name == "Agent One"

        assert config.get_agent_by_id("nonexistent") is None

    def test_get_agent_by_handle(self):
        """Test get_agent_by_handle method (case-insensitive)."""
        config = AgentsConfig(
            default_agent={"id": "default"},
            agents=[
                {
                    "id": "default",
                    "name": "Default",
                    "handles": ["assistant"],
                    "url": "http://localhost",
                    "routing": {"examples": ["help"]},
                },
                {
                    "id": "agent-1",
                    "name": "Agent One",
                    "handles": ["myhandle", "alias"],
                    "url": "http://agent1",
                    "routing": {"examples": ["test"]},
                },
            ],
        )
        # Test first handle (lowercase)
        agent = config.get_agent_by_handle("myhandle")
        assert agent is not None
        assert agent.id == "agent-1"

        # Test second handle (alias)
        agent = config.get_agent_by_handle("alias")
        assert agent is not None
        assert agent.id == "agent-1"

        # Test mixed case (should still match)
        agent = config.get_agent_by_handle("MyHandle")
        assert agent is not None
        assert agent.id == "agent-1"

        # Test uppercase
        agent = config.get_agent_by_handle("MYHANDLE")
        assert agent is not None
        assert agent.id == "agent-1"

        # Test uppercase alias
        agent = config.get_agent_by_handle("ALIAS")
        assert agent is not None
        assert agent.id == "agent-1"

        assert config.get_agent_by_handle("nonexistent") is None

    def test_get_agent_by_handle_multiple_agents(self):
        """Test get_agent_by_handle with multiple agents having different handles."""
        config = AgentsConfig(
            default_agent={"id": "default"},
            agents=[
                {
                    "id": "default",
                    "name": "Default",
                    "handles": ["assistant"],
                    "url": "http://localhost",
                    "routing": {"examples": ["help"]},
                },
                {
                    "id": "agent-1",
                    "name": "Agent One",
                    "handles": ["troubleshoot", "debug", "rca"],
                    "url": "http://agent1",
                    "routing": {"examples": ["test"]},
                },
                {
                    "id": "agent-2",
                    "name": "Agent Two",
                    "handles": ["metrics", "prometheus"],
                    "url": "http://agent2",
                    "routing": {"examples": ["test"]},
                },
            ],
        )
        # Test first agent handles
        assert config.get_agent_by_handle("troubleshoot").id == "agent-1"
        assert config.get_agent_by_handle("debug").id == "agent-1"
        assert config.get_agent_by_handle("rca").id == "agent-1"

        # Test second agent handles
        assert config.get_agent_by_handle("metrics").id == "agent-2"
        assert config.get_agent_by_handle("prometheus").id == "agent-2"

    def test_agents_array_required(self):
        """Test that agents array must have at least one agent."""
        with pytest.raises(ValueError):
            AgentsConfig(
                default_agent={"id": "nonexistent"},
                agents=[],
            )


class TestLoadAgentsConfig:
    """Test YAML config loading."""

    def test_load_valid_config(self):
        """Test loading a valid config file."""
        yaml_content = """
session:
  sticky_enabled: true
  timeout_minutes: 30
  topic_drift_threshold: 0.5

default_agent:
  id: "default-agent"

agents:
  - id: "default-agent"
    name: "Default Agent"
    handles:
      - "assistant"
    url: "http://default-agent:8080"
    protocol: "a2a"
    routing:
      priority: 100
      threshold: 0.3
      examples:
        - "general question"

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
        - "example query"
    description: "A test agent"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_agents_config(f.name)

        assert config.session.sticky_enabled is True
        assert config.default_agent.id == "default-agent"
        assert len(config.agents) == 2
        assert config.agents[1].id == "test-agent"
        assert config.agents[1].handles == ["test", "debug"]

        # Verify get_default_agent works
        default = config.get_default_agent()
        assert default.id == "default-agent"
        assert default.handles == ["assistant"]

        Path(f.name).unlink()

    def test_load_minimal_config(self):
        """Test loading minimal config with default agent in agents array."""
        yaml_content = """
default_agent:
  id: "my-default"

agents:
  - id: "my-default"
    name: "Default"
    handles:
      - "assistant"
    url: "http://default-agent:8080"
    routing:
      examples:
        - "help"
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = load_agents_config(f.name)

        assert config.session.sticky_enabled is True
        assert config.default_agent.id == "my-default"
        assert len(config.agents) == 1
        assert config.get_default_agent().handles == ["assistant"]

        Path(f.name).unlink()

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(ConfigLoadError, match="not found"):
            load_agents_config("/tmp/nonexistent_agents_config.yaml")

    def test_load_invalid_yaml(self):
        """Test loading invalid YAML raises error."""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()

            with pytest.raises(ConfigLoadError, match="Invalid YAML"):
                load_agents_config(f.name)

        Path(f.name).unlink()

    def test_load_empty_file(self):
        """Test loading empty file raises error."""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()

            with pytest.raises(ConfigLoadError, match="Empty configuration"):
                load_agents_config(f.name)

        Path(f.name).unlink()

    def test_load_invalid_schema(self):
        """Test loading file with invalid schema raises error."""
        yaml_content = """
session:
  timeout_minutes: -5
default_agent:
  id: "default"
agents:
  - id: "default"
    name: "Default"
    handles: ["assistant"]
    url: "http://localhost"
    routing:
      examples: ["help"]
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ConfigLoadError, match="validation failed"):
                load_agents_config(f.name)

        Path(f.name).unlink()

    def test_load_missing_default_agent(self):
        """Test loading file without default_agent raises error."""
        yaml_content = """
session:
  sticky_enabled: true
agents:
  - id: "test"
    name: "Test"
    handles: ["test"]
    url: "http://localhost"
    routing:
      examples: ["test"]
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ConfigLoadError, match="validation failed"):
                load_agents_config(f.name)

        Path(f.name).unlink()

    def test_load_default_agent_not_in_agents(self):
        """Test loading file where default_agent.id doesn't exist in agents."""
        yaml_content = """
default_agent:
  id: "nonexistent"

agents:
  - id: "test-agent"
    name: "Test"
    handles: ["test"]
    url: "http://localhost"
    routing:
      examples: ["test"]
"""
        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            with pytest.raises(ConfigLoadError, match="validation failed"):
                load_agents_config(f.name)

        Path(f.name).unlink()
