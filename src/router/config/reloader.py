"""Configuration hot-reload orchestrator.

Coordinates reloading of agent configuration and rebuilding the semantic router.
"""

import logging
import threading
from typing import TYPE_CHECKING

from router.config.agents import ConfigLoadError, clear_agents_config_cache, load_agents_config
from router.config.watcher import ConfigWatcher

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("router.audit")


class ConfigReloader:
    """Orchestrates hot-reloading of agent configuration.

    Watches the config file for changes and coordinates:
    1. Clearing the config cache
    2. Loading the new configuration
    3. Rebuilding the semantic router index
    4. Updating app.state atomically

    Thread-safe: uses a lock to prevent concurrent reloads.
    """

    def __init__(
        self,
        app: "FastAPI",
        config_path: str,
        debounce_seconds: float = 1.0,
    ) -> None:
        """Initialize the config reloader.

        Args:
            app: The FastAPI application instance.
            config_path: Path to the config file to watch.
            debounce_seconds: Minimum time between reload triggers.
        """
        self._app = app
        self._config_path = config_path
        self._debounce_seconds = debounce_seconds
        self._watcher: ConfigWatcher | None = None
        self._reload_lock = threading.Lock()
        self._reload_count = 0

    def start(self) -> None:
        """Start watching for config file changes."""
        if self._watcher is not None:
            logger.warning("Config reloader already started")
            return

        self._watcher = ConfigWatcher(
            config_path=self._config_path,
            on_change=self._on_config_change,
            debounce_seconds=self._debounce_seconds,
        )
        self._watcher.start()
        logger.info("Config hot-reload enabled")

    def stop(self) -> None:
        """Stop watching for config file changes."""
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
            logger.info("Config hot-reload disabled")

    def reload(self) -> bool:
        """Manually trigger a config reload.

        Returns:
            True if reload succeeded, False otherwise.
        """
        return self._on_config_change()

    def _on_config_change(self) -> bool:
        """Handle config file change event.

        Returns:
            True if reload succeeded, False otherwise.
        """
        # Use non-blocking acquire to skip if reload already in progress
        if not self._reload_lock.acquire(blocking=False):
            logger.info("Config reload already in progress, skipping")
            return False

        try:
            return self._perform_reload()
        finally:
            self._reload_lock.release()

    def _perform_reload(self) -> bool:
        """Perform the actual config reload.

        Returns:
            True if reload succeeded, False otherwise.
        """
        logger.info("Reloading agent configuration...")
        audit_logger.info(
            '{"event": "config_reload_started", "config_path": "%s"}',
            self._config_path,
        )

        # Step 1: Clear the config cache
        clear_agents_config_cache()

        # Step 2: Load and validate new configuration
        try:
            new_config = load_agents_config(self._config_path)
        except ConfigLoadError as e:
            logger.error("Failed to load new configuration: %s", e)
            audit_logger.error(
                '{"event": "config_reload_failed", "reason": "load_error", '
                '"error": "%s"}',
                str(e)[:200],
            )
            return False
        except Exception as e:
            logger.exception("Unexpected error loading configuration")
            audit_logger.error(
                '{"event": "config_reload_failed", "reason": "unexpected_error", '
                '"error": "%s"}',
                str(e)[:200],
            )
            return False

        # Step 3: Rebuild semantic router index
        semantic_router = getattr(self._app.state, "semantic_router", None)
        if semantic_router is None:
            logger.error("Semantic router not initialized, cannot reload")
            audit_logger.error(
                '{"event": "config_reload_failed", "reason": "no_semantic_router"}'
            )
            return False

        try:
            semantic_router.build_index(new_config)
        except Exception as e:
            logger.exception("Failed to rebuild semantic router index")
            audit_logger.error(
                '{"event": "config_reload_failed", "reason": "index_build_error", '
                '"error": "%s"}',
                str(e)[:200],
            )
            return False

        # Step 4: Update app.state atomically
        self._app.state.agents_config = new_config
        self._reload_count += 1

        # Log success
        agent_ids = [agent.id for agent in new_config.agents]
        logger.info(
            "Configuration reloaded successfully (reload #%d): %d agents",
            self._reload_count,
            len(new_config.agents),
        )
        for agent in new_config.agents:
            logger.info(
                "  - %s (@%s) via %s",
                agent.name,
                ", @".join(agent.handles),
                agent.protocol.value,
            )

        audit_logger.info(
            '{"event": "config_reload_success", "reload_count": %d, '
            '"agent_count": %d, "agent_ids": %s}',
            self._reload_count,
            len(new_config.agents),
            agent_ids,
        )

        return True

    @property
    def reload_count(self) -> int:
        """Get the number of successful reloads."""
        return self._reload_count

    @property
    def is_running(self) -> bool:
        """Check if the watcher is running."""
        return self._watcher is not None and self._watcher.is_running
