"""Tests for sticky session functionality."""

import time
from datetime import datetime, timedelta

from router.session import SessionState, SessionStore


class TestSessionState:
    """Tests for SessionState dataclass."""

    def test_creation(self):
        """Test basic SessionState creation."""
        state = SessionState(
            thread_id="thread-123",
            agent_id="agent-1",
            agent_handle="test",
        )
        assert state.thread_id == "thread-123"
        assert state.agent_id == "agent-1"
        assert state.agent_handle == "test"
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.last_activity, datetime)

    def test_touch_updates_last_activity(self):
        """Test that touch() updates last_activity timestamp."""
        state = SessionState(
            thread_id="thread-123",
            agent_id="agent-1",
            agent_handle="test",
        )
        original_time = state.last_activity

        # Wait a small amount to ensure time difference
        time.sleep(0.01)
        state.touch()

        assert state.last_activity > original_time

    def test_is_expired_false_for_fresh_session(self):
        """Test that fresh sessions are not expired."""
        state = SessionState(
            thread_id="thread-123",
            agent_id="agent-1",
            agent_handle="test",
        )
        assert not state.is_expired(timeout_minutes=30)

    def test_is_expired_true_for_old_session(self):
        """Test that old sessions are expired."""
        state = SessionState(
            thread_id="thread-123",
            agent_id="agent-1",
            agent_handle="test",
        )
        # Manually set last_activity to be old
        state.last_activity = datetime.now() - timedelta(minutes=60)

        assert state.is_expired(timeout_minutes=30)

    def test_is_expired_boundary(self):
        """Test expiration at the boundary."""
        state = SessionState(
            thread_id="thread-123",
            agent_id="agent-1",
            agent_handle="test",
        )
        # Set to exactly 30 minutes ago - should still be valid
        state.last_activity = datetime.now() - timedelta(minutes=29, seconds=59)
        assert not state.is_expired(timeout_minutes=30)

        # Set to just over 30 minutes ago - should be expired
        state.last_activity = datetime.now() - timedelta(minutes=30, seconds=1)
        assert state.is_expired(timeout_minutes=30)


class TestSessionStore:
    """Tests for SessionStore."""

    def test_creation_with_default_timeout(self):
        """Test SessionStore creation with default timeout."""
        store = SessionStore()
        assert store.timeout_minutes == 30

    def test_creation_with_custom_timeout(self):
        """Test SessionStore creation with custom timeout."""
        store = SessionStore(timeout_minutes=60)
        assert store.timeout_minutes == 60

    def test_set_and_get(self):
        """Test setting and getting a session."""
        store = SessionStore()

        session = store.set(
            thread_id="thread-123",
            agent_id="agent-1",
            agent_handle="test",
        )

        retrieved = store.get("thread-123")
        assert retrieved is not None
        assert retrieved.thread_id == "thread-123"
        assert retrieved.agent_id == "agent-1"
        assert retrieved == session

    def test_get_nonexistent(self):
        """Test getting a nonexistent session."""
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_get_expired_returns_none(self):
        """Test that getting an expired session returns None."""
        store = SessionStore(timeout_minutes=1)

        store.set(
            thread_id="thread-123",
            agent_id="agent-1",
            agent_handle="test",
        )

        # Manually expire the session
        session = store._sessions["thread-123"]
        session.last_activity = datetime.now() - timedelta(minutes=2)

        assert store.get("thread-123") is None
        # Session should be cleaned up
        assert "thread-123" not in store._sessions

    def test_set_replaces_existing(self):
        """Test that set replaces existing session."""
        store = SessionStore()

        store.set(
            thread_id="thread-123",
            agent_id="agent-1",
            agent_handle="test1",
        )
        store.set(
            thread_id="thread-123",
            agent_id="agent-2",
            agent_handle="test2",
        )

        retrieved = store.get("thread-123")
        assert retrieved is not None
        assert retrieved.agent_id == "agent-2"
        assert retrieved.agent_handle == "test2"

    def test_touch_updates_activity(self):
        """Test that touch updates session activity."""
        store = SessionStore()

        store.set(
            thread_id="thread-123",
            agent_id="agent-1",
            agent_handle="test",
        )

        original_time = store.get("thread-123").last_activity

        time.sleep(0.01)
        result = store.touch("thread-123")

        assert result is True
        assert store.get("thread-123").last_activity > original_time

    def test_touch_nonexistent_returns_false(self):
        """Test that touch on nonexistent session returns False."""
        store = SessionStore()
        assert store.touch("nonexistent") is False

    def test_touch_expired_returns_false(self):
        """Test that touch on expired session returns False."""
        store = SessionStore(timeout_minutes=1)

        store.set(
            thread_id="thread-123",
            agent_id="agent-1",
            agent_handle="test",
        )

        # Expire the session
        store._sessions["thread-123"].last_activity = datetime.now() - timedelta(minutes=2)

        assert store.touch("thread-123") is False
        assert "thread-123" not in store._sessions

    def test_delete(self):
        """Test deleting a session."""
        store = SessionStore()

        store.set(
            thread_id="thread-123",
            agent_id="agent-1",
            agent_handle="test",
        )

        result = store.delete("thread-123")
        assert result is True
        assert store.get("thread-123") is None

    def test_delete_nonexistent_returns_false(self):
        """Test that delete on nonexistent session returns False."""
        store = SessionStore()
        assert store.delete("nonexistent") is False

    def test_cleanup_expired(self):
        """Test cleaning up expired sessions."""
        store = SessionStore(timeout_minutes=1)

        # Create some sessions
        store.set("thread-1", "agent-1", "test1")
        store.set("thread-2", "agent-2", "test2")
        store.set("thread-3", "agent-3", "test3")

        # Expire two of them
        store._sessions["thread-1"].last_activity = datetime.now() - timedelta(minutes=2)
        store._sessions["thread-2"].last_activity = datetime.now() - timedelta(minutes=2)

        removed = store.cleanup_expired()
        assert removed == 2
        assert store.get("thread-1") is None
        assert store.get("thread-2") is None
        assert store.get("thread-3") is not None

    def test_count(self):
        """Test session count."""
        store = SessionStore()
        assert store.count() == 0

        store.set("thread-1", "agent-1", "test1")
        assert store.count() == 1

        store.set("thread-2", "agent-2", "test2")
        assert store.count() == 2

        store.delete("thread-1")
        assert store.count() == 1

    def test_clear(self):
        """Test clearing all sessions."""
        store = SessionStore()

        store.set("thread-1", "agent-1", "test1")
        store.set("thread-2", "agent-2", "test2")

        store.clear()
        assert store.count() == 0
        assert store.get("thread-1") is None
        assert store.get("thread-2") is None
