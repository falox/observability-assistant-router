"""In-memory session store with TTL expiration."""

import logging
import threading

from router.session.state import SessionState

logger = logging.getLogger(__name__)


class SessionStore:
    """Thread-safe in-memory session store with TTL expiration.

    Stores session state keyed by thread_id (AG-UI conversation identifier).
    Sessions expire after a configurable timeout period of inactivity.

    Note: This implementation is suitable for single-instance deployments.
    For horizontal scaling, consider using Redis or another distributed store.
    """

    def __init__(self, timeout_minutes: int = 30) -> None:
        """Initialize the session store.

        Args:
            timeout_minutes: Number of minutes before sessions expire.
        """
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._timeout_minutes = timeout_minutes

    @property
    def timeout_minutes(self) -> int:
        """Get the session timeout in minutes."""
        return self._timeout_minutes

    def get(self, thread_id: str) -> SessionState | None:
        """Get a session by thread ID.

        Returns None if session doesn't exist or has expired.
        Automatically cleans up expired sessions.

        Args:
            thread_id: The AG-UI thread ID (conversation identifier).

        Returns:
            SessionState if found and not expired, None otherwise.
        """
        with self._lock:
            session = self._sessions.get(thread_id)

            if session is None:
                return None

            if session.is_expired(self._timeout_minutes):
                logger.debug(
                    "Session expired (thread_id=%s, agent_id=%s, last_activity=%s)",
                    thread_id,
                    session.agent_id,
                    session.last_activity.isoformat(),
                )
                del self._sessions[thread_id]
                return None

            return session

    def set(
        self,
        thread_id: str,
        agent_id: str,
        agent_handle: str,
    ) -> SessionState:
        """Create or update a session.

        If a session already exists, it will be replaced with a new one.

        Args:
            thread_id: The AG-UI thread ID (conversation identifier).
            agent_id: The ID of the agent handling this conversation.
            agent_handle: The @mention handle of the agent.

        Returns:
            The created SessionState.
        """
        session = SessionState(
            thread_id=thread_id,
            agent_id=agent_id,
            agent_handle=agent_handle,
        )

        with self._lock:
            existing = self._sessions.get(thread_id)
            if existing:
                logger.debug(
                    "Replacing session (thread_id=%s, old_agent=%s, new_agent=%s)",
                    thread_id,
                    existing.agent_id,
                    agent_id,
                )
            else:
                logger.debug(
                    "Creating session (thread_id=%s, agent_id=%s)",
                    thread_id,
                    agent_id,
                )

            self._sessions[thread_id] = session

        return session

    def touch(self, thread_id: str) -> bool:
        """Update the last activity timestamp of a session.

        Args:
            thread_id: The AG-UI thread ID.

        Returns:
            True if session was found and updated, False if not found.
        """
        with self._lock:
            session = self._sessions.get(thread_id)
            if session is None:
                return False

            if session.is_expired(self._timeout_minutes):
                del self._sessions[thread_id]
                return False

            session.touch()
            return True

    def delete(self, thread_id: str) -> bool:
        """Delete a session.

        Args:
            thread_id: The AG-UI thread ID.

        Returns:
            True if session was deleted, False if not found.
        """
        with self._lock:
            if thread_id in self._sessions:
                logger.debug("Deleting session (thread_id=%s)", thread_id)
                del self._sessions[thread_id]
                return True
            return False

    def cleanup_expired(self) -> int:
        """Remove all expired sessions.

        This can be called periodically to clean up memory.

        Returns:
            Number of sessions removed.
        """
        expired_ids: list[str] = []

        with self._lock:
            for thread_id, session in self._sessions.items():
                if session.is_expired(self._timeout_minutes):
                    expired_ids.append(thread_id)

            for thread_id in expired_ids:
                del self._sessions[thread_id]

        if expired_ids:
            logger.info("Cleaned up %d expired sessions", len(expired_ids))

        return len(expired_ids)

    def count(self) -> int:
        """Get the number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def clear(self) -> None:
        """Remove all sessions."""
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            if count:
                logger.info("Cleared %d sessions", count)
