"""Tests for the AG-UI client passthrough functionality."""

import json

import pytest

from router.agui.client import AGUIClient


class TestParseEvent:
    """Tests for the _parse_event method - passthrough for all AG-UI events."""

    @pytest.fixture
    def client(self):
        """Create an AGUIClient with no HTTP client (not needed for parsing tests)."""
        return AGUIClient(http_client=None)

    def test_parse_run_started_event(self, client):
        """Test that RUN_STARTED events are passed through."""
        data = '{"type": "RUN_STARTED", "thread_id": "t-123", "run_id": "r-456"}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "RUN_STARTED"

    def test_parse_run_finished_event(self, client):
        """Test that RUN_FINISHED events are passed through."""
        data = '{"type": "RUN_FINISHED", "thread_id": "t-123", "run_id": "r-456"}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "RUN_FINISHED"

    def test_parse_text_message_start_event(self, client):
        """Test that TEXT_MESSAGE_START events are passed through."""
        data = '{"type": "TEXT_MESSAGE_START", "messageId": "m-123", "role": "assistant"}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "TEXT_MESSAGE_START"

    def test_parse_text_message_content_event(self, client):
        """Test that TEXT_MESSAGE_CONTENT events are passed through."""
        data = '{"type": "TEXT_MESSAGE_CONTENT", "messageId": "m-123", "delta": "Hello"}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "TEXT_MESSAGE_CONTENT"

    def test_parse_text_message_end_event(self, client):
        """Test that TEXT_MESSAGE_END events are passed through."""
        data = '{"type": "TEXT_MESSAGE_END", "messageId": "m-123"}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "TEXT_MESSAGE_END"

    def test_parse_run_error_event(self, client):
        """Test that RUN_ERROR events are passed through."""
        data = '{"type": "RUN_ERROR", "message": "Something went wrong", "code": "ERROR"}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "RUN_ERROR"

    def test_parse_tool_call_start_event(self, client):
        """Test that TOOL_CALL_START events are passed through."""
        data = '{"type": "TOOL_CALL_START", "toolCallId": "tc-123", "toolCallName": "search"}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "TOOL_CALL_START"

    def test_parse_tool_call_args_event(self, client):
        """Test that TOOL_CALL_ARGS events are passed through."""
        data = (
            '{"type": "TOOL_CALL_ARGS", "toolCallId": "tc-123", '
            '"delta": "{\\"query\\": \\"test\\"}"}'
        )
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "TOOL_CALL_ARGS"

    def test_parse_tool_call_end_event(self, client):
        """Test that TOOL_CALL_END events are passed through."""
        data = '{"type": "TOOL_CALL_END", "toolCallId": "tc-123"}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "TOOL_CALL_END"

    def test_parse_tool_call_result_event(self, client):
        """Test that TOOL_CALL_RESULT events are passed through."""
        data = '{"type": "TOOL_CALL_RESULT", "toolCallId": "tc-123", "result": "success"}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "TOOL_CALL_RESULT"

    def test_parse_state_delta_event(self, client):
        """Test that STATE_DELTA events are passed through via BaseEvent."""
        data = '{"type": "STATE_DELTA", "delta": {"key": "value"}}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "STATE_DELTA"

    def test_parse_state_snapshot_event(self, client):
        """Test that STATE_SNAPSHOT events are passed through via BaseEvent."""
        data = '{"type": "STATE_SNAPSHOT", "snapshot": {"key": "value"}}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "STATE_SNAPSHOT"

    def test_parse_step_started_event(self, client):
        """Test that STEP_STARTED events are passed through via BaseEvent."""
        data = '{"type": "STEP_STARTED", "stepName": "analyze"}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "STEP_STARTED"

    def test_parse_step_finished_event(self, client):
        """Test that STEP_FINISHED events are passed through via BaseEvent."""
        data = '{"type": "STEP_FINISHED", "stepName": "analyze"}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "STEP_FINISHED"

    def test_parse_custom_event(self, client):
        """Test that CUSTOM events are passed through via BaseEvent."""
        data = '{"type": "CUSTOM", "name": "my_event", "value": {"foo": "bar"}}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "CUSTOM"

    def test_parse_messages_snapshot_event(self, client):
        """Test that MESSAGES_SNAPSHOT events are passed through via BaseEvent."""
        data = '{"type": "MESSAGES_SNAPSHOT", "messages": []}'
        event = client._parse_event(data, None)

        assert event is not None
        assert event.type.value == "MESSAGES_SNAPSHOT"

    def test_parse_thinking_events(self, client):
        """Test that THINKING_* events are passed through via BaseEvent."""
        events_data = [
            '{"type": "THINKING_START"}',
            '{"type": "THINKING_TEXT_MESSAGE_START", "messageId": "m-1", "role": "assistant"}',
            '{"type": "THINKING_TEXT_MESSAGE_CONTENT", "messageId": "m-1", "delta": "thinking..."}',
            '{"type": "THINKING_TEXT_MESSAGE_END", "messageId": "m-1"}',
            '{"type": "THINKING_END"}',
        ]
        expected_types = [
            "THINKING_START",
            "THINKING_TEXT_MESSAGE_START",
            "THINKING_TEXT_MESSAGE_CONTENT",
            "THINKING_TEXT_MESSAGE_END",
            "THINKING_END",
        ]

        for data, expected_type in zip(events_data, expected_types, strict=True):
            event = client._parse_event(data, None)
            assert event is not None, f"Failed to parse {expected_type}"
            assert event.type.value == expected_type

    def test_preserves_all_fields(self, client):
        """Test that all fields are preserved in passthrough."""
        data = json.dumps(
            {
                "type": "TEXT_MESSAGE_CONTENT",
                "messageId": "m-123",
                "delta": "Hello world",
                "customField": "should be preserved",
                "nested": {"key": "value"},
            }
        )
        event = client._parse_event(data, None)

        assert event is not None
        # Check that extra fields are preserved
        dumped = event.model_dump()
        assert "customField" in dumped
        assert dumped["customField"] == "should be preserved"
        assert dumped["nested"] == {"key": "value"}


class TestParseEventFallback:
    """Tests for event type fallback from SSE header."""

    @pytest.fixture
    def client(self):
        """Create an AGUIClient with no HTTP client."""
        return AGUIClient(http_client=None)

    def test_uses_type_from_data(self, client):
        """Test that type from JSON data is used when present."""
        data = '{"type": "RUN_STARTED"}'
        event = client._parse_event(data, "SHOULD_NOT_USE")

        assert event is not None
        assert event.type.value == "RUN_STARTED"

    def test_falls_back_to_event_header(self, client):
        """Test that SSE event: header is used when type missing from data."""
        data = '{"messageId": "m-123", "delta": "Hello"}'
        event = client._parse_event(data, "TEXT_MESSAGE_CONTENT")

        assert event is not None
        assert event.type.value == "TEXT_MESSAGE_CONTENT"

    def test_returns_none_when_no_type(self, client):
        """Test that None is returned when no type is available."""
        data = '{"messageId": "m-123", "delta": "Hello"}'
        event = client._parse_event(data, None)

        assert event is None


class TestParseEventErrorHandling:
    """Tests for error handling in _parse_event."""

    @pytest.fixture
    def client(self):
        """Create an AGUIClient with no HTTP client."""
        return AGUIClient(http_client=None)

    def test_returns_none_for_empty_data(self, client):
        """Test that empty data returns None."""
        assert client._parse_event("", None) is None

    def test_returns_none_for_done_marker(self, client):
        """Test that [DONE] marker returns None."""
        assert client._parse_event("[DONE]", None) is None

    def test_returns_none_for_invalid_json(self, client):
        """Test that invalid JSON returns None."""
        event = client._parse_event("not valid json", None)
        assert event is None

    def test_returns_none_for_missing_type(self, client):
        """Test that missing type field returns None."""
        event = client._parse_event('{"foo": "bar"}', None)
        assert event is None


class TestBaseEventSerialization:
    """Tests for BaseEvent serialization in passthrough."""

    @pytest.fixture
    def client(self):
        """Create an AGUIClient with no HTTP client."""
        return AGUIClient(http_client=None)

    def test_round_trip_serialization(self, client):
        """Test that events can be serialized back to JSON correctly."""
        original_data = {
            "type": "TEXT_MESSAGE_CONTENT",
            "messageId": "m-123",
            "delta": "Hello world",
        }
        data = json.dumps(original_data)
        event = client._parse_event(data, None)

        assert event is not None

        # Serialize back to JSON
        json_str = event.model_dump_json(by_alias=True, exclude_none=True)
        parsed = json.loads(json_str)

        # Verify key fields are preserved
        assert parsed["type"] == "TEXT_MESSAGE_CONTENT"
        assert parsed["messageId"] == "m-123"
        assert parsed["delta"] == "Hello world"

    def test_excludes_none_values(self, client):
        """Test that None values are excluded from serialized JSON."""
        data = '{"type": "RUN_STARTED"}'
        event = client._parse_event(data, None)

        json_str = event.model_dump_json(by_alias=True, exclude_none=True)

        # timestamp and raw_event are None by default, should be excluded
        assert "timestamp" not in json_str or "null" not in json_str
