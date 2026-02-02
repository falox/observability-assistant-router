"""File watcher for hot-reloading agent configuration.

Supports both regular file changes and symlink updates (for Kubernetes ConfigMaps).
"""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """Handler for config file changes with debouncing.

    Handles both direct file modifications and symlink updates
    (which is how Kubernetes ConfigMaps are updated).
    """

    def __init__(
        self,
        config_path: Path,
        on_change: Callable[[], None],
        debounce_seconds: float = 1.0,
    ) -> None:
        """Initialize the config file handler.

        Args:
            config_path: Path to the config file to watch.
            on_change: Callback function to invoke when config changes.
            debounce_seconds: Minimum time between reload triggers.
        """
        super().__init__()
        self._config_path = config_path.resolve()
        self._config_name = config_path.name
        self._on_change = on_change
        self._debounce_seconds = debounce_seconds
        self._last_trigger = 0.0
        self._lock = threading.Lock()
        self._pending_reload = False
        self._timer: threading.Timer | None = None

    def _trigger_reload(self) -> None:
        """Trigger the reload callback with debouncing."""
        with self._lock:
            now = time.time()
            time_since_last = now - self._last_trigger

            if time_since_last < self._debounce_seconds:
                # Schedule a delayed reload if not already scheduled
                if not self._pending_reload:
                    self._pending_reload = True
                    delay = self._debounce_seconds - time_since_last
                    self._timer = threading.Timer(delay, self._execute_reload)
                    self._timer.start()
                return

            self._execute_reload_internal()

    def _execute_reload(self) -> None:
        """Execute the scheduled reload."""
        with self._lock:
            self._pending_reload = False
            self._execute_reload_internal()

    def _execute_reload_internal(self) -> None:
        """Internal reload execution (must be called with lock held)."""
        self._last_trigger = time.time()
        logger.info("Config file change detected, triggering reload")
        try:
            self._on_change()
        except Exception:
            logger.exception("Error in config reload callback")

    def _is_config_event(self, event: FileSystemEvent) -> bool:
        """Check if the event is related to the config file.

        Handles both direct file changes and symlink updates used by
        Kubernetes ConfigMaps (which update via atomic symlink swap).
        """
        if event.is_directory:
            return False

        event_path = Path(event.src_path).resolve()

        # Direct match
        if event_path == self._config_path:
            return True

        # Check if it's the same filename in the watched directory
        # (handles symlink targets and ..data directory used by ConfigMaps)
        if event_path.name == self._config_name:
            return True

        # Check for Kubernetes ConfigMap symlink patterns
        # ConfigMaps use ..data -> ..timestamp_directory structure
        if "..data" in str(event_path) or event_path.name.startswith(".."):
            # Check if the config file exists and was modified
            if self._config_path.exists():
                return True

        return False

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if self._is_config_event(event):
            logger.debug("Config modified event: %s", event.src_path)
            self._trigger_reload()

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events (symlink swaps appear as creates)."""
        if self._is_config_event(event):
            logger.debug("Config created event: %s", event.src_path)
            self._trigger_reload()

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events (atomic file updates)."""
        if self._is_config_event(event):
            logger.debug("Config moved event: %s", event.src_path)
            self._trigger_reload()

    def cancel_pending(self) -> None:
        """Cancel any pending reload timer."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending_reload = False


class ConfigWatcher:
    """Watches the agent config file for changes and triggers reloads.

    Works with both regular files and Kubernetes ConfigMap volumes
    (which use symlinks that get atomically swapped on updates).
    """

    def __init__(
        self,
        config_path: str | Path,
        on_change: Callable[[], None],
        debounce_seconds: float = 1.0,
    ) -> None:
        """Initialize the config watcher.

        Args:
            config_path: Path to the config file to watch.
            on_change: Callback function to invoke when config changes.
            debounce_seconds: Minimum time between reload triggers.
        """
        self._config_path = Path(config_path).resolve()
        self._on_change = on_change
        self._debounce_seconds = debounce_seconds
        self._observer: Observer | None = None
        self._handler: ConfigFileHandler | None = None

    def start(self) -> None:
        """Start watching for config file changes."""
        if self._observer is not None:
            logger.warning("Config watcher already started")
            return

        # Watch the parent directory (required to catch symlink updates)
        watch_dir = self._config_path.parent

        if not watch_dir.exists():
            logger.warning(
                "Config directory does not exist, watcher not started: %s",
                watch_dir,
            )
            return

        self._handler = ConfigFileHandler(
            config_path=self._config_path,
            on_change=self._on_change,
            debounce_seconds=self._debounce_seconds,
        )

        self._observer = Observer()
        # recursive=True is needed for Kubernetes ConfigMap symlink structure
        self._observer.schedule(self._handler, str(watch_dir), recursive=True)
        self._observer.start()

        logger.info(
            "Config watcher started, monitoring: %s (debounce: %.1fs)",
            self._config_path,
            self._debounce_seconds,
        )

    def stop(self) -> None:
        """Stop watching for config file changes."""
        if self._handler is not None:
            self._handler.cancel_pending()
            self._handler = None

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
            logger.info("Config watcher stopped")

    @property
    def is_running(self) -> bool:
        """Check if the watcher is running."""
        return self._observer is not None and self._observer.is_alive()
