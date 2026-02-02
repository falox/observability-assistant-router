"""Agent configuration loader and Pydantic models."""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import AnyHttpUrl, BaseModel, Field, field_validator, model_validator

from router.config.settings import get_settings


class AgentProtocol(str, Enum):
    """Supported agent protocols."""

    A2A = "a2a"
    AG_UI = "ag-ui"


class SessionConfig(BaseModel):
    """Sticky session configuration."""

    sticky_enabled: bool = True
    timeout_minutes: int = Field(default=30, ge=1)
    topic_drift_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class DefaultAgentConfig(BaseModel):
    """Reference to the default agent by ID."""

    id: Annotated[str, Field(min_length=1, max_length=100)]


class AgentRoutingConfig(BaseModel):
    """Routing configuration for an agent."""

    priority: int = Field(default=1, ge=1)
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    examples: Annotated[
        list[Annotated[str, Field(max_length=500)]],
        Field(default_factory=list, max_length=100),
    ]


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    id: Annotated[str, Field(min_length=1, max_length=100)]
    name: Annotated[str, Field(max_length=200)]
    handles: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=50)]],
        Field(min_length=1, max_length=10),
    ]
    url: AnyHttpUrl
    protocol: AgentProtocol = AgentProtocol.A2A
    routing: AgentRoutingConfig | None = None
    description: Annotated[str, Field(max_length=1000)] = ""

    @field_validator("handles")
    @classmethod
    def handles_lowercase(cls, v: list[str]) -> list[str]:
        """Ensure all handles are lowercase for case-insensitive matching."""
        return [h.lower() for h in v]


class AgentsConfig(BaseModel):
    """Root configuration for all agents."""

    session: SessionConfig = Field(default_factory=SessionConfig)
    default_agent: DefaultAgentConfig
    agents: list[AgentConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_default_agent_exists(self) -> "AgentsConfig":
        """Validate that the default agent ID references an existing agent."""
        default_id = self.default_agent.id
        agent_ids = [agent.id for agent in self.agents]
        if default_id not in agent_ids:
            raise ValueError(
                f"default_agent.id '{default_id}' not found in agents. "
                f"Available agent IDs: {agent_ids}"
            )
        return self

    def get_agent_by_id(self, agent_id: str) -> AgentConfig | None:
        """Get an agent by its ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    def get_agent_by_handle(self, handle: str) -> AgentConfig | None:
        """Get an agent by any of its handles (case-insensitive)."""
        handle_lower = handle.lower()
        for agent in self.agents:
            if handle_lower in agent.handles:
                return agent
        return None

    def get_default_agent(self) -> AgentConfig:
        """Get the default agent configuration.

        Returns:
            The AgentConfig for the default agent.

        Raises:
            ValueError: If default agent not found (should not happen after validation).
        """
        agent = self.get_agent_by_id(self.default_agent.id)
        if agent is None:
            raise ValueError(f"Default agent '{self.default_agent.id}' not found")
        return agent

    def is_default_agent(self, agent: AgentConfig) -> bool:
        """Check if the given agent is the default agent."""
        return agent.id == self.default_agent.id


class ConfigLoadError(Exception):
    """Raised when configuration loading fails."""

    pass


def _validate_config_path(path: Path, allowed_dirs: list[str]) -> Path:
    """Validate and canonicalize config path to prevent path traversal.

    Args:
        path: The path to validate.
        allowed_dirs: List of allowed directory prefixes.

    Returns:
        Canonicalized (resolved) path.

    Raises:
        ConfigLoadError: If path is outside allowed directories.
    """
    resolved = path.resolve()

    # Check if resolved path starts with any allowed directory
    for allowed_dir in allowed_dirs:
        allowed_resolved = Path(allowed_dir).resolve()
        try:
            resolved.relative_to(allowed_resolved)
            return resolved
        except ValueError:
            continue

    raise ConfigLoadError(
        f"Configuration path '{resolved}' is outside allowed directories: {allowed_dirs}"
    )


# Allowed directories for configuration files (security: path traversal prevention)
# Note: /tmp included for testing; in production, ConfigMap mounts to /config
ALLOWED_CONFIG_DIRS = ["/config", "/app/config", "/tmp", "config", "."]


def load_agents_config(config_path: str | Path | None = None) -> AgentsConfig:
    """Load and validate agents configuration from YAML file.

    Args:
        config_path: Path to the YAML config file. If None, uses settings.

    Returns:
        Validated AgentsConfig instance.

    Raises:
        ConfigLoadError: If file cannot be read, validation fails, or path traversal detected.
    """
    if config_path is None:
        config_path = get_settings().config_path

    path = Path(config_path)

    # Security: Validate path to prevent traversal attacks
    path = _validate_config_path(path, ALLOWED_CONFIG_DIRS)

    if not path.exists():
        raise ConfigLoadError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Invalid YAML in {path}: {e}") from e
    except OSError as e:
        raise ConfigLoadError(f"Failed to read {path}: {e}") from e

    if raw_config is None:
        raise ConfigLoadError(f"Empty configuration file: {path}")

    try:
        return AgentsConfig.model_validate(raw_config)
    except ValueError as e:
        raise ConfigLoadError(f"Configuration validation failed: {e}") from e


@lru_cache
def get_agents_config() -> AgentsConfig:
    """Get cached agents configuration instance."""
    return load_agents_config()


def clear_agents_config_cache() -> None:
    """Clear the cached agents configuration.

    Call this before reloading configuration to ensure fresh data is loaded.
    """
    get_agents_config.cache_clear()
