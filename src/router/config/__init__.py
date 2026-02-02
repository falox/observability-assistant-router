"""Configuration module for the router."""

from router.config.agents import (
    AgentConfig,
    AgentProtocol,
    AgentRoutingConfig,
    AgentsConfig,
    ConfigLoadError,
    DefaultAgentConfig,
    SessionConfig,
    clear_agents_config_cache,
    get_agents_config,
    load_agents_config,
)
from router.config.reloader import ConfigReloader
from router.config.settings import Settings, get_settings
from router.config.watcher import ConfigWatcher

__all__ = [
    "AgentConfig",
    "AgentProtocol",
    "AgentRoutingConfig",
    "AgentsConfig",
    "ConfigLoadError",
    "ConfigReloader",
    "ConfigWatcher",
    "DefaultAgentConfig",
    "SessionConfig",
    "Settings",
    "clear_agents_config_cache",
    "get_agents_config",
    "get_settings",
    "load_agents_config",
]
