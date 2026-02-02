"""Tests for the observability module."""

import json
import logging
import time
from datetime import datetime
from unittest.mock import patch

import pytest

from router.observability import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    BufferedMessage,
    StreamBuffer,
    StreamFrame,
)
from router.observability.audit import configure_audit_logging


class TestStreamFrame:
    """Tests for StreamFrame dataclass."""

    def test_creation(self):
        """Test StreamFrame creation with all fields."""
        frame = StreamFrame(
            event_type="TEXT_MESSAGE_CONTENT",
            data={"delta": "Hello"},
            timestamp=1234567890.0,
            sequence_num=5,
        )

        assert frame.event_type == "TEXT_MESSAGE_CONTENT"
        assert frame.data == {"delta": "Hello"}
        assert frame.timestamp == 1234567890.0
        assert frame.sequence_num == 5


class TestBufferedMessage:
    """Tests for BufferedMessage dataclass."""

    def test_creation_defaults(self):
        """Test BufferedMessage with default values."""
        msg = BufferedMessage(
            message_id="msg-1",
            thread_id="thread-1",
            run_id="run-1",
        )

        assert msg.message_id == "msg-1"
        assert msg.thread_id == "thread-1"
        assert msg.run_id == "run-1"
        assert msg.frames == []
        assert msg.complete is False
        assert msg.start_time is None
        assert msg.completion_time is None
        assert msg.accumulated_content == ""

    def test_creation_full(self):
        """Test BufferedMessage with all fields."""
        frame = StreamFrame(
            event_type="TEXT_MESSAGE_START",
            data={},
            timestamp=1234567890.0,
            sequence_num=0,
        )

        msg = BufferedMessage(
            message_id="msg-1",
            thread_id="thread-1",
            run_id="run-1",
            frames=[frame],
            complete=True,
            start_time=1234567890.0,
            completion_time=1234567891.0,
            accumulated_content="Hello, world!",
        )

        assert len(msg.frames) == 1
        assert msg.complete is True
        assert msg.accumulated_content == "Hello, world!"


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_creation(self):
        """Test AuditEvent creation."""
        now = datetime.now()
        event = AuditEvent(
            event_type=AuditEventType.REQUEST_RECEIVED,
            timestamp=now,
            request_id="req-123",
            thread_id="thread-456",
            metadata={"message_count": 5},
        )

        assert event.event_type == AuditEventType.REQUEST_RECEIVED
        assert event.timestamp == now
        assert event.request_id == "req-123"
        assert event.thread_id == "thread-456"
        assert event.metadata == {"message_count": 5}

    def test_to_dict(self):
        """Test AuditEvent serialization to dict."""
        now = datetime.now()
        event = AuditEvent(
            event_type=AuditEventType.ROUTING_DECISION,
            timestamp=now,
            request_id="req-123",
            thread_id="thread-456",
            metadata={"agent_id": "agent-1", "score": 0.95},
        )

        result = event.to_dict()

        assert result["event_type"] == "routing_decision"
        assert result["timestamp"] == now.isoformat()
        assert result["request_id"] == "req-123"
        assert result["thread_id"] == "thread-456"
        assert result["agent_id"] == "agent-1"
        assert result["score"] == 0.95


class TestStreamBuffer:
    """Tests for StreamBuffer class."""

    def test_creation(self):
        """Test StreamBuffer initialization."""
        buffer = StreamBuffer(
            thread_id="thread-1",
            run_id="run-1",
            request_id="req-1",
        )

        assert buffer.message is None
        assert buffer.is_complete is False

    def test_process_message_start(self):
        """Test processing TEXT_MESSAGE_START event."""
        buffer = StreamBuffer(
            thread_id="thread-1",
            run_id="run-1",
            request_id="req-1",
        )

        event = {
            "event": "TEXT_MESSAGE_START",
            "data": json.dumps({"messageId": "msg-1", "role": "assistant"}),
        }

        buffer._process_event(event)

        assert buffer.message is not None
        assert buffer.message.message_id == "msg-1"
        assert buffer.message.thread_id == "thread-1"
        assert buffer.message.start_time is not None
        assert len(buffer.message.frames) == 1

    def test_process_message_content(self):
        """Test processing TEXT_MESSAGE_CONTENT event."""
        buffer = StreamBuffer(
            thread_id="thread-1",
            run_id="run-1",
            request_id="req-1",
        )

        # Start message
        buffer._process_event(
            {
                "event": "TEXT_MESSAGE_START",
                "data": json.dumps({"messageId": "msg-1"}),
            }
        )

        # Add content
        buffer._process_event(
            {
                "event": "TEXT_MESSAGE_CONTENT",
                "data": json.dumps({"messageId": "msg-1", "delta": "Hello "}),
            }
        )

        buffer._process_event(
            {
                "event": "TEXT_MESSAGE_CONTENT",
                "data": json.dumps({"messageId": "msg-1", "delta": "world!"}),
            }
        )

        assert buffer.message.accumulated_content == "Hello world!"
        assert len(buffer.message.frames) == 3

    def test_process_message_end(self):
        """Test processing TEXT_MESSAGE_END event."""
        buffer = StreamBuffer(
            thread_id="thread-1",
            run_id="run-1",
            request_id="req-1",
        )

        # Start message
        buffer._process_event(
            {
                "event": "TEXT_MESSAGE_START",
                "data": json.dumps({"messageId": "msg-1"}),
            }
        )

        # Add content
        buffer._process_event(
            {
                "event": "TEXT_MESSAGE_CONTENT",
                "data": json.dumps({"delta": "Hello"}),
            }
        )

        # End message
        buffer._process_event(
            {
                "event": "TEXT_MESSAGE_END",
                "data": json.dumps({"messageId": "msg-1"}),
            }
        )

        assert buffer.is_complete is True
        assert buffer.message.completion_time is not None
        assert len(buffer.message.frames) == 3

    def test_content_size_limit(self):
        """Test that content is truncated at max size."""
        buffer = StreamBuffer(
            thread_id="thread-1",
            run_id="run-1",
            request_id="req-1",
            max_content_size=10,
        )

        # Start message
        buffer._process_event(
            {
                "event": "TEXT_MESSAGE_START",
                "data": json.dumps({"messageId": "msg-1"}),
            }
        )

        # Add content that exceeds limit
        buffer._process_event(
            {
                "event": "TEXT_MESSAGE_CONTENT",
                "data": json.dumps({"delta": "This is a very long message that exceeds the limit"}),
            }
        )

        assert len(buffer.message.accumulated_content) == 10
        assert buffer.message.accumulated_content == "This is a "

    def test_get_stats(self):
        """Test getting buffer statistics."""
        buffer = StreamBuffer(
            thread_id="thread-1",
            run_id="run-1",
            request_id="req-1",
        )

        # Process a complete message
        buffer._process_event(
            {
                "event": "TEXT_MESSAGE_START",
                "data": json.dumps({"messageId": "msg-1"}),
            }
        )
        buffer._process_event(
            {
                "event": "TEXT_MESSAGE_CONTENT",
                "data": json.dumps({"delta": "Hello"}),
            }
        )
        buffer._process_event(
            {
                "event": "TEXT_MESSAGE_END",
                "data": json.dumps({}),
            }
        )

        stats = buffer.get_stats()

        assert stats["request_id"] == "req-1"
        assert stats["thread_id"] == "thread-1"
        assert stats["total_frames"] == 3
        assert stats["has_message"] is True
        assert stats["is_complete"] is True
        assert stats["message_id"] == "msg-1"
        assert stats["frame_count"] == 3
        assert "duration_ms" in stats

    @pytest.mark.asyncio
    async def test_process_stream_passthrough(self):
        """Test that process_stream yields all events."""
        buffer = StreamBuffer(
            thread_id="thread-1",
            run_id="run-1",
            request_id="req-1",
        )

        async def event_generator():
            yield {
                "event": "TEXT_MESSAGE_START",
                "data": json.dumps({"messageId": "msg-1"}),
            }
            yield {
                "event": "TEXT_MESSAGE_CONTENT",
                "data": json.dumps({"delta": "Hello"}),
            }
            yield {
                "event": "TEXT_MESSAGE_END",
                "data": json.dumps({}),
            }

        events = []
        async for event in buffer.process_stream(event_generator()):
            events.append(event)

        assert len(events) == 3
        assert buffer.is_complete is True
        assert buffer.message.accumulated_content == "Hello"


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_creation(self):
        """Test AuditLogger initialization."""
        logger = AuditLogger(
            request_id="req-123",
            thread_id="thread-456",
            enabled=True,
        )

        assert logger._request_id == "req-123"
        assert logger._thread_id == "thread-456"
        assert logger._enabled is True

    def test_disabled_logger_does_not_emit(self):
        """Test that disabled logger doesn't emit events."""
        logger = AuditLogger(
            request_id="req-123",
            thread_id="thread-456",
            enabled=False,
        )

        with patch.object(logging.getLogger("router.audit"), "info") as mock_info:
            logger.log_request_received(message_count=5)
            mock_info.assert_not_called()

    def test_log_request_received(self):
        """Test logging request received event."""
        logger = AuditLogger(
            request_id="req-123",
            thread_id="thread-456",
        )

        with patch.object(logging.getLogger("router.audit"), "info") as mock_info:
            logger.log_request_received(
                message_count=5,
                has_authorization=True,
                user_message_preview="Hello, I need help",
            )

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            data = json.loads(call_args)

            assert data["event_type"] == "request_received"
            assert data["request_id"] == "req-123"
            assert data["thread_id"] == "thread-456"
            assert data["message_count"] == 5
            assert data["has_authorization"] is True
            assert data["user_message_preview"] == "Hello, I need help"

    def test_log_routing_decision(self):
        """Test logging routing decision event."""
        logger = AuditLogger(
            request_id="req-123",
            thread_id="thread-456",
        )

        with patch.object(logging.getLogger("router.audit"), "info") as mock_info:
            logger.log_routing_decision(
                agent_id="agent-1",
                agent_name="Troubleshooting Agent",
                routing_method="semantic",
                confidence_score=0.95,
            )

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            data = json.loads(call_args)

            assert data["event_type"] == "routing_decision"
            assert data["agent_id"] == "agent-1"
            assert data["agent_name"] == "Troubleshooting Agent"
            assert data["routing_method"] == "semantic"
            assert data["confidence_score"] == 0.95

    def test_log_agent_forwarded(self):
        """Test logging agent forwarded event."""
        logger = AuditLogger(
            request_id="req-123",
            thread_id="thread-456",
        )

        with patch.object(logging.getLogger("router.audit"), "info") as mock_info:
            logger.log_agent_forwarded(
                agent_id="agent-1",
                agent_protocol="a2a",
                attempt_number=2,
            )

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            data = json.loads(call_args)

            assert data["event_type"] == "agent_forwarded"
            assert data["agent_id"] == "agent-1"
            assert data["agent_protocol"] == "a2a"
            assert data["attempt_number"] == 2

    def test_log_agent_error(self):
        """Test logging agent error event."""
        logger = AuditLogger(
            request_id="req-123",
            thread_id="thread-456",
        )

        with patch.object(logging.getLogger("router.audit"), "info") as mock_info:
            logger.log_agent_error(
                agent_id="agent-1",
                error_message="Connection refused",
                status_code=503,
                is_retryable=True,
                attempt_number=3,
            )

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            data = json.loads(call_args)

            assert data["event_type"] == "agent_error"
            assert data["agent_id"] == "agent-1"
            assert data["error_message"] == "Connection refused"
            assert data["status_code"] == 503
            assert data["is_retryable"] is True
            assert data["attempt_number"] == 3

    def test_log_fallback_triggered(self):
        """Test logging fallback triggered event."""
        logger = AuditLogger(
            request_id="req-123",
            thread_id="thread-456",
        )

        with patch.object(logging.getLogger("router.audit"), "info") as mock_info:
            logger.log_fallback_triggered(
                original_agent_id="agent-1",
                fallback_agent_id="default",
                reason="Agent unavailable after 3 attempts",
            )

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            data = json.loads(call_args)

            assert data["event_type"] == "fallback_triggered"
            assert data["original_agent_id"] == "agent-1"
            assert data["fallback_agent_id"] == "default"
            assert data["reason"] == "Agent unavailable after 3 attempts"

    def test_log_message_complete(self):
        """Test logging message complete event."""
        logger = AuditLogger(
            request_id="req-123",
            thread_id="thread-456",
        )

        msg = BufferedMessage(
            message_id="msg-1",
            thread_id="thread-456",
            run_id="run-1",
            frames=[StreamFrame("START", {}, time.time(), 0)],
            complete=True,
            start_time=time.time() - 1.5,
            completion_time=time.time(),
            accumulated_content="Hello, world!",
        )

        with patch.object(logging.getLogger("router.audit"), "info") as mock_info:
            logger.log_message_complete(msg)

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            data = json.loads(call_args)

            assert data["event_type"] == "message_complete"
            assert data["message_id"] == "msg-1"
            assert data["content_length"] == 13
            assert data["frame_count"] == 1
            assert "duration_ms" in data

    def test_log_session_event(self):
        """Test logging session lifecycle events."""
        logger = AuditLogger(
            request_id="req-123",
            thread_id="thread-456",
        )

        with patch.object(logging.getLogger("router.audit"), "info") as mock_info:
            logger.log_session_event(
                action="created",
                agent_id="agent-1",
                reason="new_conversation",
            )

            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            data = json.loads(call_args)

            assert data["event_type"] == "session_created"
            assert data["action"] == "created"
            assert data["agent_id"] == "agent-1"
            assert data["reason"] == "new_conversation"

    def test_error_message_truncation(self):
        """Test that long error messages are truncated."""
        logger = AuditLogger(
            request_id="req-123",
            thread_id="thread-456",
        )

        long_error = "A" * 500  # 500 character error

        with patch.object(logging.getLogger("router.audit"), "info") as mock_info:
            logger.log_agent_error(
                agent_id="agent-1",
                error_message=long_error,
            )

            call_args = mock_info.call_args[0][0]
            data = json.loads(call_args)

            # Should be truncated to 200 chars
            assert len(data["error_message"]) == 200

    def test_user_message_preview_truncation(self):
        """Test that user message preview is truncated."""
        logger = AuditLogger(
            request_id="req-123",
            thread_id="thread-456",
        )

        long_message = "B" * 200  # 200 character message

        with patch.object(logging.getLogger("router.audit"), "info") as mock_info:
            logger.log_request_received(
                message_count=1,
                user_message_preview=long_message,
            )

            call_args = mock_info.call_args[0][0]
            data = json.loads(call_args)

            # Should be truncated to 100 chars
            assert len(data["user_message_preview"]) == 100


class TestConfigureAuditLogging:
    """Tests for audit logging configuration."""

    def test_configure_sets_level(self):
        """Test that configure_audit_logging sets the log level."""
        audit_log = logging.getLogger("router.audit")

        # Clear existing handlers
        audit_log.handlers.clear()

        configure_audit_logging("DEBUG")

        assert audit_log.level == logging.DEBUG

    def test_configure_prevents_propagation(self):
        """Test that audit logger doesn't propagate to root."""
        audit_log = logging.getLogger("router.audit")

        # Clear existing handlers
        audit_log.handlers.clear()

        configure_audit_logging("INFO")

        assert audit_log.propagate is False
