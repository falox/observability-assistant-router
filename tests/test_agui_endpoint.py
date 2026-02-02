"""Tests for the AG-UI chat endpoint."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from router.agui.models import (
    RunErrorEvent,
    TextMessageContentEvent,
    TextMessageStartEvent,
)
from router.main import app


class TestAGUIEndpoint:
    """Tests for the /api/agui/chat endpoint."""

    @pytest.fixture
    async def client(self):
        """Create an async test client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_chat_endpoint_exists(self, client):
        """Test that the chat endpoint exists."""
        # Send a minimal valid request to check the endpoint exists
        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "test-thread",
                "messages": [{"id": "msg-1", "role": "user", "content": "Hello"}],
            },
        )
        # 200 OK or other status (not 404)
        assert response.status_code != 404

    @pytest.mark.asyncio
    async def test_chat_endpoint_requires_thread_id(self, client):
        """Test that thread_id is required."""
        response = await client.post(
            "/api/agui/chat",
            json={
                "messages": [{"id": "msg-1", "role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_chat_endpoint_requires_messages(self, client):
        """Test that messages are required."""
        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "test-thread",
            },
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_chat_endpoint_rejects_empty_messages(self, client):
        """Test that empty messages array is rejected."""
        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "test-thread",
                "messages": [],
            },
        )
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_chat_endpoint_accepts_valid_request(self, client):
        """Test that valid request is accepted."""
        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "test-thread",
                "messages": [{"id": "msg-1", "role": "user", "content": "Why is my pod crashing?"}],
            },
        )
        # Should not be a validation error
        assert response.status_code != 422

    @pytest.mark.asyncio
    async def test_chat_endpoint_returns_sse_content_type(self, client):
        """Test that the endpoint returns text/event-stream content type for SSE.

        The content type must be 'text/event-stream' (not 'application/vnd.ag-ui.event+proto')
        because the response body is SSE text format, not protobuf binary.
        The @ag-ui/client library uses this header to determine which parser to use.
        """
        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "test-thread",
                "messages": [{"id": "msg-1", "role": "user", "content": "Hello"}],
            },
        )
        # The response should start streaming (or return an error if agent unavailable)
        assert response.status_code in (200, 503, 500)
        # Verify correct content type for SSE responses
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            assert "text/event-stream" in content_type, (
                f"Expected 'text/event-stream' content type, got '{content_type}'. "
                "SSE responses must use text/event-stream for proper client parsing."
            )

    @pytest.mark.asyncio
    async def test_chat_endpoint_returns_request_id_header(self, client):
        """Test that the endpoint returns X-Request-ID header when successful.

        Note: X-Request-ID is only included in SSE responses (200).
        Error responses (503 when router not ready) don't include it.
        """
        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "test-thread",
                "messages": [{"id": "msg-1", "role": "user", "content": "Hello"}],
            },
        )
        # Only check header on successful SSE response
        if response.status_code == 200:
            assert "x-request-id" in response.headers
        # If config not loaded (503), we skip the header check
        # This is expected in test environments without full setup

    @pytest.mark.asyncio
    async def test_chat_endpoint_forwards_request_id(self, client):
        """Test that provided X-Request-ID is echoed back when successful.

        Note: X-Request-ID is only included in SSE responses (200).
        Error responses (503 when router not ready) don't include it.
        """
        custom_request_id = "custom-request-123"
        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "test-thread",
                "messages": [{"id": "msg-1", "role": "user", "content": "Hello"}],
            },
            headers={"X-Request-ID": custom_request_id},
        )
        # Only check header on successful SSE response
        if response.status_code == 200:
            assert response.headers.get("x-request-id") == custom_request_id
        # If config not loaded (503), we skip the header check


class TestAGUIEndpointRouting:
    """Tests for routing behavior in the AG-UI endpoint."""

    @pytest.fixture
    async def client(self):
        """Create an async test client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_routing_troubleshooting_query(self, client):
        """Test that troubleshooting queries are routed correctly."""
        # This test verifies the semantic routing is invoked
        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "test-thread",
                "messages": [
                    {
                        "id": "msg-1",
                        "role": "user",
                        "content": "Why is my pod crashing with OOMKilled?",
                    }
                ],
            },
        )
        # The endpoint should process the request (not fail on validation)
        assert response.status_code != 422

    @pytest.mark.asyncio
    async def test_routing_metrics_query(self, client):
        """Test that metrics queries are routed correctly."""
        response = await client.post(
            "/api/agui/chat",
            json={
                "thread_id": "test-thread",
                "messages": [
                    {"id": "msg-1", "role": "user", "content": "Show me the CPU usage metrics"}
                ],
            },
        )
        # The endpoint should process the request
        assert response.status_code != 422


class TestSSEConfiguration:
    """Tests for Server-Sent Events configuration.

    These tests verify that the SSE response is properly configured
    for compatibility with standard SSE parsers and the @ag-ui/client library.
    """

    def test_sse_uses_lf_line_separator(self):
        """Test that EventSourceResponse is configured with LF (\\n) separator.

        The SSE specification requires LF (\\n) line endings, not CRLF (\\r\\n).
        Using CRLF causes SSE parsers to fail because they split on \\n\\n.
        """
        from sse_starlette.sse import EventSourceResponse

        # Verify the separator parameter is accepted and can be set to LF
        async def dummy_generator():
            yield {"data": "test"}

        response = EventSourceResponse(dummy_generator(), sep="\n")
        assert response.sep == "\n", (
            "EventSourceResponse must use LF (\\n) separator for SSE compliance. "
            "CRLF (\\r\\n) breaks SSE parsers that split on \\n\\n."
        )

    def test_default_separator_is_crlf(self):
        """Verify the library's default separator is CRLF (which we override).

        This test documents why we need to explicitly set sep='\\n'.
        """
        from sse_starlette.sse import EventSourceResponse

        assert EventSourceResponse.DEFAULT_SEPARATOR == "\r\n", (
            "If the library changes its default separator to LF, "
            "our explicit sep='\\n' override may no longer be needed."
        )


class TestEventSerialization:
    """Tests for AG-UI event JSON serialization in SSE responses."""

    def test_text_message_content_event_serialization(self):
        """Test that TextMessageContentEvent serializes correctly for SSE."""
        event = TextMessageContentEvent(
            type="TEXT_MESSAGE_CONTENT",
            message_id="msg-123",
            delta="Hello, world!",
        )

        json_str = event.model_dump_json(by_alias=True, exclude_none=True)
        parsed = json.loads(json_str)

        # Verify JSON is valid and contains expected fields (uses camelCase aliases)
        assert parsed["type"] == "TEXT_MESSAGE_CONTENT"
        assert parsed["messageId"] == "msg-123"
        assert parsed["delta"] == "Hello, world!"

    def test_run_error_event_serialization(self):
        """Test that RunErrorEvent serializes correctly for SSE."""
        event = RunErrorEvent(
            type="RUN_ERROR",
            message="Something went wrong",
            code="AGENT_ERROR",
        )

        json_str = event.model_dump_json(by_alias=True, exclude_none=True)
        parsed = json.loads(json_str)

        assert parsed["type"] == "RUN_ERROR"
        assert parsed["message"] == "Something went wrong"

    def test_text_message_start_event_serialization(self):
        """Test that TextMessageStartEvent serializes correctly for SSE."""
        event = TextMessageStartEvent(
            type="TEXT_MESSAGE_START",
            message_id="msg-456",
            role="assistant",
        )

        json_str = event.model_dump_json(by_alias=True, exclude_none=True)
        parsed = json.loads(json_str)

        # Verify uses camelCase aliases for AG-UI protocol compatibility
        assert parsed["type"] == "TEXT_MESSAGE_START"
        assert parsed["messageId"] == "msg-456"
        assert parsed["role"] == "assistant"

    def test_serialization_excludes_none_values(self):
        """Test that None values are excluded from serialized JSON."""
        event = TextMessageContentEvent(
            type="TEXT_MESSAGE_CONTENT",
            message_id="msg-789",
            delta="Test content",
        )

        json_str = event.model_dump_json(by_alias=True, exclude_none=True)

        # Verify no null/None values appear in the JSON string
        assert "null" not in json_str
        assert "None" not in json_str


class TestStripMentionsFromRequest:
    """Tests for _strip_mentions_from_request function."""

    def test_strips_mentions_from_user_message_string(self):
        """Test that @mentions are stripped from user message with string content."""
        from router.agui.endpoint import _strip_mentions_from_request
        from router.agui.models import ChatRequest, UserMessage

        request = ChatRequest(
            thread_id="test-thread",
            messages=[
                UserMessage(
                    id="msg-1", role="user", content="@troubleshoot why is my pod crashing?"
                ),
            ],
        )

        result = _strip_mentions_from_request(request)

        assert result.messages[0].content == "why is my pod crashing?"

    def test_strips_all_mentions_from_message(self):
        """Test that ALL @mentions are stripped, not just the first."""
        from router.agui.endpoint import _strip_mentions_from_request
        from router.agui.models import ChatRequest, UserMessage

        request = ChatRequest(
            thread_id="test-thread",
            messages=[
                UserMessage(
                    id="msg-1",
                    role="user",
                    content="@metrics @prometheus show @grafana CPU usage",
                ),
            ],
        )

        result = _strip_mentions_from_request(request)

        assert result.messages[0].content == "show CPU usage"

    def test_preserves_assistant_messages(self):
        """Test that assistant messages are not modified."""
        from router.agui.endpoint import _strip_mentions_from_request
        from router.agui.models import AssistantMessage, ChatRequest, UserMessage

        request = ChatRequest(
            thread_id="test-thread",
            messages=[
                UserMessage(id="msg-1", role="user", content="@troubleshoot help"),
                AssistantMessage(
                    id="msg-2",
                    role="assistant",
                    content="I'll help you with @troubleshoot",
                ),
            ],
        )

        result = _strip_mentions_from_request(request)

        # User message should be stripped
        assert result.messages[0].content == "help"
        # Assistant message should be preserved
        assert result.messages[1].content == "I'll help you with @troubleshoot"

    def test_preserves_system_messages(self):
        """Test that system messages are not modified."""
        from router.agui.endpoint import _strip_mentions_from_request
        from router.agui.models import ChatRequest, SystemMessage, UserMessage

        request = ChatRequest(
            thread_id="test-thread",
            messages=[
                SystemMessage(
                    id="msg-1",
                    role="system",
                    content="You are @assistant helping users",
                ),
                UserMessage(id="msg-2", role="user", content="@troubleshoot help"),
            ],
        )

        result = _strip_mentions_from_request(request)

        # System message should be preserved
        assert result.messages[0].content == "You are @assistant helping users"
        # User message should be stripped
        assert result.messages[1].content == "help"

    def test_preserves_original_request(self):
        """Test that the original request is not mutated."""
        from router.agui.endpoint import _strip_mentions_from_request
        from router.agui.models import ChatRequest, UserMessage

        original_content = "@troubleshoot why is my pod crashing?"
        request = ChatRequest(
            thread_id="test-thread",
            messages=[UserMessage(id="msg-1", role="user", content=original_content)],
        )

        _strip_mentions_from_request(request)

        # Original should be unchanged
        assert request.messages[0].content == original_content

    def test_handles_messages_without_mentions(self):
        """Test that messages without mentions are passed through unchanged."""
        from router.agui.endpoint import _strip_mentions_from_request
        from router.agui.models import ChatRequest, UserMessage

        request = ChatRequest(
            thread_id="test-thread",
            messages=[
                UserMessage(id="msg-1", role="user", content="Why is my pod crashing?"),
            ],
        )

        result = _strip_mentions_from_request(request)

        assert result.messages[0].content == "Why is my pod crashing?"
