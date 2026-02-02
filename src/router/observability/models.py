"""Data models for observability components.

This module defines dataclasses for stream buffering and audit logging,
providing structured types for observability data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class AuditEventType(str, Enum):
    """Types of audit events logged by the router."""

    REQUEST_RECEIVED = "request_received"
    ROUTING_DECISION = "routing_decision"
    AGENT_FORWARDED = "agent_forwarded"
    STREAM_STARTED = "stream_started"
    STREAM_CHUNK = "stream_chunk"
    MESSAGE_COMPLETE = "message_complete"
    SESSION_CREATED = "session_created"
    SESSION_UPDATED = "session_updated"
    SESSION_EXPIRED = "session_expired"
    SESSION_DELETED = "session_deleted"
    AGENT_ERROR = "agent_error"
    FALLBACK_TRIGGERED = "fallback_triggered"


@dataclass
class StreamFrame:
    """A single frame in a streaming message.

    Represents one SSE event in a streaming response, used for
    tracking and reassembling complete messages.
    """

    event_type: str
    """The AG-UI event type (e.g., 'TEXT_MESSAGE_CONTENT')."""

    data: dict[str, Any]
    """The event payload as a dictionary."""

    timestamp: float
    """Unix timestamp when this frame was received."""

    sequence_num: int
    """Sequence number for ordering frames."""


@dataclass
class BufferedMessage:
    """A complete message reassembled from stream frames.

    Holds all frames for a message along with metadata about
    the message completion status and accumulated content.
    """

    message_id: str
    """The AG-UI message ID."""

    thread_id: str
    """The conversation thread ID."""

    run_id: str
    """The run ID for this request."""

    frames: list[StreamFrame] = field(default_factory=list)
    """All frames in this message, in order."""

    complete: bool = False
    """Whether the message has been completed (TEXT_MESSAGE_END received)."""

    start_time: float | None = None
    """Unix timestamp when the message started."""

    completion_time: float | None = None
    """Unix timestamp when the message completed."""

    accumulated_content: str = ""
    """Full text content accumulated from all content frames."""


@dataclass
class AuditEvent:
    """A structured audit event for logging.

    Provides a consistent format for audit trail entries that can
    be serialized to JSON for structured logging.
    """

    event_type: AuditEventType
    """The type of audit event."""

    timestamp: datetime
    """When the event occurred."""

    request_id: str
    """The request ID for correlation."""

    thread_id: str
    """The conversation thread ID."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Event-specific metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the audit event to a dictionary for JSON serialization.

        Returns:
            Dictionary representation of the event.
        """
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
            "thread_id": self.thread_id,
            **self.metadata,
        }
