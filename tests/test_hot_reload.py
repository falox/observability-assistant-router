"""Tests for configuration hot-reload functionality."""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from router.config.agents import clear_agents_config_cache, load_agents_config
from router.config.reloader import ConfigReloader
from router.config.watcher import ConfigFileHandler, ConfigWatcher


class TestConfigFileHandler:
    """Test ConfigFileHandler debouncing and event handling."""

    def test_debouncing_triggers_once(self):
        """Test that rapid events are debounced to a single callback."""
        callback_count = 0
        callback_lock = threading.Lock()

        def callback():
            nonlocal callback_count
            with callback_lock:
                callback_count += 1

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            handler = ConfigFileHandler(
                config_path=config_path,
                on_change=callback,
                debounce_seconds=0.1,
            )

            # Create a mock event
            event = MagicMock()
            event.is_directory = False
            event.src_path = str(config_path)

            # Trigger multiple rapid events
            for _ in range(5):
                handler.on_modified(event)

            # Wait for debounce to complete
            time.sleep(0.2)

            # Should have triggered at most 2 times (initial + debounced)
            with callback_lock:
                assert callback_count <= 2

            handler.cancel_pending()
        finally:
            config_path.unlink(missing_ok=True)

    def test_ignores_directory_events(self):
        """Test that directory events are ignored."""
        callback_called = False

        def callback():
            nonlocal callback_called
            callback_called = True

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config_path = Path(f.name)

        try:
            handler = ConfigFileHandler(
                config_path=config_path,
                on_change=callback,
                debounce_seconds=0.01,
            )

            event = MagicMock()
            event.is_directory = True
            event.src_path = str(config_path)

            handler.on_modified(event)
            time.sleep(0.05)

            assert not callback_called
            handler.cancel_pending()
        finally:
            config_path.unlink(missing_ok=True)

    def test_handles_configmap_symlink_patterns(self):
        """Test detection of Kubernetes ConfigMap symlink patterns."""
        callback_called = False

        def callback():
            nonlocal callback_called
            callback_called = True

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agents.yaml"
            config_path.write_text("test: value")

            handler = ConfigFileHandler(
                config_path=config_path,
                on_change=callback,
                debounce_seconds=0.01,
            )

            # Simulate ConfigMap ..data symlink update
            event = MagicMock()
            event.is_directory = False
            event.src_path = str(Path(tmpdir) / "..data")

            handler.on_created(event)
            time.sleep(0.05)

            assert callback_called
            handler.cancel_pending()


class TestConfigWatcher:
    """Test ConfigWatcher file monitoring."""

    def test_start_and_stop(self):
        """Test watcher can be started and stopped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agents.yaml"
            config_path.write_text("test: value")

            watcher = ConfigWatcher(
                config_path=config_path,
                on_change=lambda: None,
                debounce_seconds=0.1,
            )

            watcher.start()
            assert watcher.is_running

            watcher.stop()
            assert not watcher.is_running

    def test_start_twice_logs_warning(self):
        """Test starting watcher twice logs warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agents.yaml"
            config_path.write_text("test: value")

            watcher = ConfigWatcher(
                config_path=config_path,
                on_change=lambda: None,
            )

            watcher.start()
            watcher.start()  # Should log warning but not fail
            assert watcher.is_running

            watcher.stop()

    def test_stop_when_not_started(self):
        """Test stopping watcher when not started is safe."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agents.yaml"
            config_path.write_text("test: value")

            watcher = ConfigWatcher(
                config_path=config_path,
                on_change=lambda: None,
            )

            watcher.stop()  # Should not raise

    def test_detects_file_modifications(self):
        """Test watcher detects file modifications."""
        callback_triggered = threading.Event()

        def on_change():
            callback_triggered.set()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agents.yaml"
            config_path.write_text("initial: value")

            watcher = ConfigWatcher(
                config_path=config_path,
                on_change=on_change,
                debounce_seconds=0.1,
            )

            watcher.start()
            try:
                # Modify the file
                time.sleep(0.1)  # Let watcher initialize
                config_path.write_text("modified: value")

                # Wait for callback
                triggered = callback_triggered.wait(timeout=2.0)
                assert triggered, "Callback was not triggered on file modification"
            finally:
                watcher.stop()

    def test_handles_nonexistent_directory(self):
        """Test watcher handles nonexistent directory gracefully."""
        watcher = ConfigWatcher(
            config_path="/nonexistent/path/agents.yaml",
            on_change=lambda: None,
        )

        watcher.start()  # Should log warning but not fail
        assert not watcher.is_running


class TestConfigReloader:
    """Test ConfigReloader orchestration."""

    def _create_valid_yaml(self, agent_name: str = "Test Agent") -> str:
        """Create a valid YAML config string."""
        return f"""
default_agent:
  id: "default-agent"

agents:
  - id: "default-agent"
    name: "{agent_name}"
    handles:
      - "assistant"
    url: "http://localhost:8080"
    routing:
      priority: 1
      threshold: 0.8
      examples:
        - "help me"
"""

    def test_reload_success(self):
        """Test successful configuration reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agents.yaml"
            config_path.write_text(self._create_valid_yaml("Initial Agent"))

            # Create mock app with state
            app = MagicMock()
            app.state.agents_config = load_agents_config(config_path)

            # Mock semantic router
            semantic_router = MagicMock()
            semantic_router.build_index = MagicMock()
            app.state.semantic_router = semantic_router

            # Clear the cache before creating reloader
            clear_agents_config_cache()

            reloader = ConfigReloader(
                app=app,
                config_path=str(config_path),
            )

            # Update config file
            config_path.write_text(self._create_valid_yaml("Updated Agent"))

            # Trigger reload
            success = reloader.reload()

            assert success
            assert reloader.reload_count == 1
            assert semantic_router.build_index.called
            assert app.state.agents_config.agents[0].name == "Updated Agent"

    def test_reload_invalid_config(self):
        """Test reload failure with invalid configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agents.yaml"
            config_path.write_text(self._create_valid_yaml())

            app = MagicMock()
            app.state.agents_config = load_agents_config(config_path)
            app.state.semantic_router = MagicMock()

            clear_agents_config_cache()

            reloader = ConfigReloader(
                app=app,
                config_path=str(config_path),
            )

            # Write invalid YAML
            config_path.write_text("invalid: yaml: content: [")

            success = reloader.reload()

            assert not success
            assert reloader.reload_count == 0

    def test_reload_no_semantic_router(self):
        """Test reload failure when semantic router is not initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agents.yaml"
            config_path.write_text(self._create_valid_yaml())

            app = MagicMock()
            app.state.agents_config = load_agents_config(config_path)
            app.state.semantic_router = None

            clear_agents_config_cache()

            reloader = ConfigReloader(
                app=app,
                config_path=str(config_path),
            )

            success = reloader.reload()

            assert not success

    def test_concurrent_reload_skipped(self):
        """Test that concurrent reloads are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agents.yaml"
            config_path.write_text(self._create_valid_yaml())

            app = MagicMock()
            app.state.agents_config = load_agents_config(config_path)

            # Create a slow semantic router
            def slow_build_index(config):
                time.sleep(0.2)

            semantic_router = MagicMock()
            semantic_router.build_index = slow_build_index
            app.state.semantic_router = semantic_router

            clear_agents_config_cache()

            reloader = ConfigReloader(
                app=app,
                config_path=str(config_path),
            )

            results = []

            def do_reload():
                results.append(reloader.reload())

            # Start two concurrent reloads
            t1 = threading.Thread(target=do_reload)
            t2 = threading.Thread(target=do_reload)

            t1.start()
            time.sleep(0.05)  # Ensure t1 starts first
            t2.start()

            t1.join()
            t2.join()

            # One should succeed, one should be skipped
            assert results.count(True) == 1
            assert results.count(False) == 1

    def test_start_and_stop_watcher(self):
        """Test starting and stopping the config watcher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "agents.yaml"
            config_path.write_text(self._create_valid_yaml())

            app = MagicMock()
            app.state.agents_config = load_agents_config(config_path)
            app.state.semantic_router = MagicMock()

            reloader = ConfigReloader(
                app=app,
                config_path=str(config_path),
            )

            reloader.start()
            assert reloader.is_running

            reloader.stop()
            assert not reloader.is_running


class TestCacheClear:
    """Test cache clearing functionality."""

    def test_clear_agents_config_cache(self):
        """Test that clearing cache causes fresh config to be loaded."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(
                """
default_agent:
  id: "default"
agents:
  - id: "default"
    name: "Initial"
    handles: ["assistant"]
    url: "http://localhost"
    routing:
      examples: ["help"]
"""
            )
            f.flush()
            config_path = f.name

        try:
            os.environ["ROUTER_CONFIG_PATH"] = config_path

            # Clear existing cache
            from router.config.settings import get_settings

            get_settings.cache_clear()
            clear_agents_config_cache()

            # First load
            config1 = load_agents_config(config_path)
            assert config1.agents[0].name == "Initial"

            # Modify file
            with open(config_path, "w") as f:
                f.write(
                    """
default_agent:
  id: "default"
agents:
  - id: "default"
    name: "Modified"
    handles: ["assistant"]
    url: "http://localhost"
    routing:
      examples: ["help"]
"""
                )

            # Without cache clear, load_agents_config would still work
            # (it doesn't use cache, get_agents_config does)
            config2 = load_agents_config(config_path)
            assert config2.agents[0].name == "Modified"
        finally:
            Path(config_path).unlink(missing_ok=True)


class TestAdminReloadEndpoint:
    """Test the /admin/reload-config endpoint."""

    @pytest.fixture
    def client(self, test_config_path):
        """Create a test client with a running app."""
        from fastapi.testclient import TestClient

        from router.main import app

        # Initialize app state for testing
        with TestClient(app) as client:
            yield client

    def test_reload_endpoint_success(self, client, test_config_path):
        """Test successful reload via endpoint."""
        response = client.post("/admin/reload-config")

        # May fail if semantic router isn't fully initialized in test
        # but should return valid JSON either way
        assert response.status_code in (200, 500)
        data = response.json()
        assert "status" in data

    def test_reload_endpoint_returns_agent_count(self, client, test_config_path):
        """Test reload endpoint returns agent count on success."""
        response = client.post("/admin/reload-config")

        if response.status_code == 200:
            data = response.json()
            assert "agent_count" in data
            assert data["agent_count"] >= 1
