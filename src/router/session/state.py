"""Session state dataclass for sticky sessions."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SessionState:
    """Represents the state of a conversation session.

    Sessions track which agent a conversation is "stuck" to, enabling
    sticky routing that keeps a conversation with the same agent until
    topic drift, user override, or timeout occurs.
    """

    thread_id: str
    agent_id: str
    agent_handle: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()

    def is_expired(self, timeout_minutes: int) -> bool:
        """Check if session has expired based on timeout.

        Args:
            timeout_minutes: Number of minutes before session expires.

        Returns:
            True if session is expired, False otherwise.
        """
        from datetime import timedelta

        expiry_time = self.last_activity + timedelta(minutes=timeout_minutes)
        return datetime.now() > expiry_time
