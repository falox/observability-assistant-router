"""Structured audit logging for the router.

This module provides the AuditLogger class that emits structured JSON
audit events for compliance, debugging, and analytics purposes.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

from router.observability.models import AuditEvent, AuditEventType, BufferedMessage

# Dedicated audit logger - separate from application logs
audit_logger = logging.getLogger("router.audit")


class AuditLogger:
    """Structured audit logger for the router.

    Emits JSON-formatted audit events to a dedicated logger for:
    - Compliance and audit trails
    - Debugging and troubleshooting
    - Analytics and monitoring

    All audit events include:
    - event_type: The type of event
    - timestamp: ISO 8601 timestamp
    - request_id: Request correlation ID
    - thread_id: Conversation thread ID
    - Additional event-specific metadata

    Usage:
        audit = AuditLogger(request_id="req-123", thread_id="thread-456")
        audit.log_request_received(message_count=5)
        audit.log_routing_decision(agent_id="agent-1", method="semantic", score=0.95)
    """

    def __init__(
        self,
        request_id: str,
        thread_id: str,
        enabled: bool = True,
    ) -> None:
        """Initialize the audit logger.

        Args:
            request_id: The request ID for correlation.
            thread_id: The conversation thread ID.
            enabled: Whether audit logging is enabled.
        """
        self._request_id = request_id
        self._thread_id = thread_id
        self._enabled = enabled

    def _emit(self, event: AuditEvent) -> None:
        """Emit an audit event to the logger.

        Args:
            event: The audit event to log.
        """
        if not self._enabled:
            return

        try:
            audit_logger.info(json.dumps(event.to_dict(), default=str))
        except Exception as e:
            # Don't let audit logging failures affect request processing
            logging.getLogger(__name__).warning("Failed to emit audit event: %s", e)

    def _create_event(
        self,
        event_type: AuditEventType,
        **metadata: Any,
    ) -> AuditEvent:
        """Create an audit event with common fields.

        Args:
            event_type: The type of audit event.
            **metadata: Event-specific metadata.

        Returns:
            The constructed AuditEvent.
        """
        return AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(UTC),
            request_id=self._request_id,
            thread_id=self._thread_id,
            metadata=metadata,
        )

    def log_request_received(
        self,
        message_count: int,
        has_authorization: bool = False,
        user_message_preview: str | None = None,
    ) -> None:
        """Log that a chat request was received.

        Args:
            message_count: Number of messages in the request.
            has_authorization: Whether an Authorization header was present.
            user_message_preview: Truncated preview of the user message (max 100 chars).
        """
        metadata: dict[str, Any] = {
            "message_count": message_count,
            "has_authorization": has_authorization,
        }

        if user_message_preview:
            # Truncate for audit log
            metadata["user_message_preview"] = user_message_preview[:100]

        self._emit(self._create_event(AuditEventType.REQUEST_RECEIVED, **metadata))

    def log_routing_decision(
        self,
        agent_id: str,
        agent_name: str,
        routing_method: str,
        confidence_score: float | None = None,
        topic_drift_detected: bool = False,
    ) -> None:
        """Log a routing decision.

        Args:
            agent_id: The selected agent's ID.
            agent_name: The selected agent's name.
            routing_method: How the agent was selected (semantic, mention, sticky, etc.).
            confidence_score: The confidence/similarity score if applicable.
            topic_drift_detected: Whether topic drift triggered re-routing.
        """
        metadata: dict[str, Any] = {
            "agent_id": agent_id,
            "agent_name": agent_name,
            "routing_method": routing_method,
        }

        if confidence_score is not None:
            metadata["confidence_score"] = round(confidence_score, 4)

        if topic_drift_detected:
            metadata["topic_drift_detected"] = True

        self._emit(self._create_event(AuditEventType.ROUTING_DECISION, **metadata))

    def log_agent_forwarded(
        self,
        agent_id: str,
        agent_protocol: str,
        attempt_number: int = 1,
    ) -> None:
        """Log that a request was forwarded to an agent.

        Args:
            agent_id: The agent's ID.
            agent_protocol: The protocol used (ag-ui or a2a).
            attempt_number: The attempt number (1 for first try).
        """
        self._emit(
            self._create_event(
                AuditEventType.AGENT_FORWARDED,
                agent_id=agent_id,
                agent_protocol=agent_protocol,
                attempt_number=attempt_number,
            )
        )

    def log_stream_started(self, run_id: str) -> None:
        """Log that streaming has started.

        Args:
            run_id: The run ID for this stream.
        """
        self._emit(
            self._create_event(
                AuditEventType.STREAM_STARTED,
                run_id=run_id,
            )
        )

    def log_stream_chunk(
        self,
        message_id: str,
        chunk_number: int,
        delta_length: int,
    ) -> None:
        """Log a stream chunk (for detailed debugging only).

        Note: This generates high volume logs. Consider enabling only at DEBUG level.

        Args:
            message_id: The message ID.
            chunk_number: The chunk sequence number.
            delta_length: Length of the content delta.
        """
        # Only log chunks at DEBUG level to avoid overwhelming the audit log
        if not audit_logger.isEnabledFor(logging.DEBUG):
            return

        self._emit(
            self._create_event(
                AuditEventType.STREAM_CHUNK,
                message_id=message_id,
                chunk_number=chunk_number,
                delta_length=delta_length,
            )
        )

    def log_message_complete(
        self,
        message: BufferedMessage,
    ) -> None:
        """Log that a message has been completely received.

        Args:
            message: The completed buffered message.
        """
        duration_ms = None
        if message.start_time and message.completion_time:
            duration_ms = round((message.completion_time - message.start_time) * 1000, 2)

        self._emit(
            self._create_event(
                AuditEventType.MESSAGE_COMPLETE,
                message_id=message.message_id,
                content_length=len(message.accumulated_content),
                frame_count=len(message.frames),
                duration_ms=duration_ms,
            )
        )

    def log_session_event(
        self,
        action: str,
        agent_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Log a session lifecycle event.

        Args:
            action: The action (created, updated, expired, deleted).
            agent_id: The agent ID associated with the session.
            reason: The reason for the action (e.g., timeout, drift, override).
        """
        event_type_map = {
            "created": AuditEventType.SESSION_CREATED,
            "updated": AuditEventType.SESSION_UPDATED,
            "expired": AuditEventType.SESSION_EXPIRED,
            "deleted": AuditEventType.SESSION_DELETED,
        }

        event_type = event_type_map.get(action, AuditEventType.SESSION_UPDATED)

        metadata: dict[str, Any] = {"action": action}
        if agent_id:
            metadata["agent_id"] = agent_id
        if reason:
            metadata["reason"] = reason

        self._emit(self._create_event(event_type, **metadata))

    def log_agent_error(
        self,
        agent_id: str,
        error_message: str,
        status_code: int | None = None,
        is_retryable: bool = False,
        attempt_number: int = 1,
    ) -> None:
        """Log an agent error.

        Args:
            agent_id: The agent's ID.
            error_message: The error message (sanitized, no credentials).
            status_code: The HTTP status code if applicable.
            is_retryable: Whether the error is retryable.
            attempt_number: The attempt number when the error occurred.
        """
        # Sanitize error message - truncate and remove potential sensitive data
        sanitized_msg = error_message[:200] if error_message else "Unknown error"

        metadata: dict[str, Any] = {
            "agent_id": agent_id,
            "error_message": sanitized_msg,
            "is_retryable": is_retryable,
            "attempt_number": attempt_number,
        }

        if status_code is not None:
            metadata["status_code"] = status_code

        self._emit(self._create_event(AuditEventType.AGENT_ERROR, **metadata))

    def log_fallback_triggered(
        self,
        original_agent_id: str,
        fallback_agent_id: str,
        reason: str,
    ) -> None:
        """Log that fallback to another agent was triggered.

        Args:
            original_agent_id: The original agent that failed.
            fallback_agent_id: The fallback agent being used.
            reason: The reason for the fallback.
        """
        self._emit(
            self._create_event(
                AuditEventType.FALLBACK_TRIGGERED,
                original_agent_id=original_agent_id,
                fallback_agent_id=fallback_agent_id,
                reason=reason[:200] if reason else "Unknown",
            )
        )


def configure_audit_logging(level: str = "INFO") -> None:
    """Configure the audit logger with JSON formatting.

    This sets up a dedicated handler for the audit logger that outputs
    structured JSON logs suitable for ingestion by log aggregation systems.

    Args:
        level: The logging level for audit events.
    """
    audit_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Only add handler if not already configured
    if not audit_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Simple format - the message is already JSON
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        audit_logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output
    audit_logger.propagate = False
