"""Tests for AG-UI models."""

import pytest
from pydantic import ValidationError

from router.agui.models import (
    AssistantMessage,
    ChatRequest,
    Context,
    EventType,
    RunStartedEvent,
    SystemMessage,
    TextMessageContentEvent,
    TextMessageStartEvent,
    Tool,
    UserMessage,
    inject_display_name,
)


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_minimal_request(self):
        """Test creating a minimal valid request."""
        request = ChatRequest(
            thread_id="thread-123",
            messages=[UserMessage(id="msg-1", role="user", content="Hello, world!")],
        )
        assert request.thread_id == "thread-123"
        assert len(request.messages) == 1
        assert request.tools == []
        assert request.context == []
        assert request.state is None

    def test_full_request(self):
        """Test creating a full request with all fields."""
        request = ChatRequest(
            thread_id="thread-456",
            messages=[
                UserMessage(id="msg-1", role="user", content="What is the weather?"),
                AssistantMessage(id="msg-2", role="assistant", content="I can help with that."),
            ],
            tools=[
                Tool(
                    name="get_weather",
                    description="Get current weather",
                    parameters={"type": "object", "properties": {}},
                )
            ],
            context=[Context(description="User location", value="San Francisco")],
            state={"conversation_step": 1},
        )
        assert request.thread_id == "thread-456"
        assert len(request.messages) == 2
        assert len(request.tools) == 1
        assert len(request.context) == 1
        assert request.state == {"conversation_step": 1}

    def test_thread_id_required(self):
        """Test that thread_id is required."""
        with pytest.raises(ValidationError, match="threadId"):
            ChatRequest(
                messages=[UserMessage(id="msg-1", role="user", content="Hello")],
            )

    def test_thread_id_not_empty(self):
        """Test that thread_id cannot be empty."""
        with pytest.raises(ValidationError, match="thread_id"):
            ChatRequest(
                thread_id="",
                messages=[UserMessage(id="msg-1", role="user", content="Hello")],
            )

    def test_messages_required(self):
        """Test that messages are required."""
        with pytest.raises(ValidationError, match="messages"):
            ChatRequest(thread_id="thread-123")

    def test_messages_not_empty(self):
        """Test that messages list cannot be empty."""
        with pytest.raises(ValidationError, match="messages"):
            ChatRequest(
                thread_id="thread-123",
                messages=[],
            )

    def test_thread_id_max_length(self):
        """Test that thread_id has a maximum length."""
        long_id = "x" * 101
        with pytest.raises(ValidationError, match="thread_id"):
            ChatRequest(
                thread_id=long_id,
                messages=[UserMessage(id="msg-1", role="user", content="Hello")],
            )


class TestMessageRoles:
    """Tests for different message roles in ChatRequest."""

    def test_user_message(self):
        """Test user message creation."""
        request = ChatRequest(
            thread_id="thread-123",
            messages=[UserMessage(id="msg-1", role="user", content="User message")],
        )
        assert request.messages[0].role == "user"

    def test_assistant_message(self):
        """Test assistant message creation."""
        request = ChatRequest(
            thread_id="thread-123",
            messages=[AssistantMessage(id="msg-1", role="assistant", content="Assistant message")],
        )
        assert request.messages[0].role == "assistant"

    def test_system_message(self):
        """Test system message creation."""
        request = ChatRequest(
            thread_id="thread-123",
            messages=[SystemMessage(id="msg-1", role="system", content="System instructions")],
        )
        assert request.messages[0].role == "system"

    def test_mixed_messages(self):
        """Test request with mixed message roles."""
        request = ChatRequest(
            thread_id="thread-123",
            messages=[
                SystemMessage(id="msg-1", role="system", content="You are helpful."),
                UserMessage(id="msg-2", role="user", content="Help me."),
                AssistantMessage(id="msg-3", role="assistant", content="Sure!"),
            ],
        )
        assert len(request.messages) == 3
        assert request.messages[0].role == "system"
        assert request.messages[1].role == "user"
        assert request.messages[2].role == "assistant"


class TestInjectDisplayName:
    """Tests for inject_display_name function."""

    def test_injects_display_name_for_run_started(self):
        """Test that displayName is injected into RUN_STARTED event."""
        event = RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id="thread-123",
            run_id="run-456",
        )
        result = inject_display_name(event, "Prometheus Expert")

        # Verify the event was modified
        assert result is not event  # New object created
        event_data = result.model_dump(by_alias=True)
        assert event_data.get("displayName") == "Prometheus Expert"
        assert event_data.get("threadId") == "thread-123"
        assert event_data.get("runId") == "run-456"

    def test_preserves_original_event_fields(self):
        """Test that original event fields are preserved when injecting displayName."""
        event = RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id="thread-789",
            run_id="run-abc",
        )
        result = inject_display_name(event, "Loki Agent")

        event_data = result.model_dump(by_alias=True)
        assert event_data.get("type") == "RUN_STARTED"
        assert event_data.get("threadId") == "thread-789"
        assert event_data.get("runId") == "run-abc"
        assert event_data.get("displayName") == "Loki Agent"

    def test_skips_non_run_started_events(self):
        """Test that non-RUN_STARTED events are not modified."""
        event = TextMessageContentEvent(
            type=EventType.TEXT_MESSAGE_CONTENT,
            message_id="msg-123",
            delta="Some content",
        )
        result = inject_display_name(event, "Agent Name")

        # Should return original event unchanged
        assert result is event

    def test_skips_text_message_start_events(self):
        """Test that TEXT_MESSAGE_START events are not modified."""
        event = TextMessageStartEvent(
            type=EventType.TEXT_MESSAGE_START,
            message_id="msg-123",
            role="assistant",
        )
        result = inject_display_name(event, "Agent Name")

        # Should return original event unchanged
        assert result is event

    def test_skips_when_display_name_is_none(self):
        """Test that events are not modified when displayName is None."""
        event = RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id="thread-123",
            run_id="run-456",
        )
        result = inject_display_name(event, None)

        # Should return original event unchanged
        assert result is event

    def test_handles_empty_display_name(self):
        """Test that empty string displayName is still injected."""
        event = RunStartedEvent(
            type=EventType.RUN_STARTED,
            thread_id="thread-123",
            run_id="run-456",
        )
        result = inject_display_name(event, "")

        event_data = result.model_dump(by_alias=True)
        assert event_data.get("displayName") == ""
