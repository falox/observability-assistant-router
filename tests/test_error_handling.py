"""Test error handling, retry, and fallback behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from router.agents.proxy import AgentProxy, AgentProxyError
from router.agents.retry import RetryConfig
from router.agui.client import AGUIClientError
from router.agui.models import (
    ChatRequest,
    EventType,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    UserMessage,
)
from router.config import AgentConfig, AgentProtocol, AgentRoutingConfig


class TestAgentProxyError:
    """Test AgentProxyError exception class."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = AgentProxyError("Test error")

        assert str(error) == "Test error"
        assert error.agent_id is None
        assert error.agent_name is None
        assert error.attempts == 1
        assert error.is_retryable is False

    def test_error_with_all_attributes(self):
        """Test error with all attributes."""
        error = AgentProxyError(
            "Test error",
            agent_id="test-id",
            agent_name="Test Agent",
            attempts=3,
            is_retryable=True,
        )

        assert str(error) == "Test error"
        assert error.agent_id == "test-id"
        assert error.agent_name == "Test Agent"
        assert error.attempts == 3
        assert error.is_retryable is True


class TestAgentProxyRetry:
    """Test AgentProxy retry behavior."""

    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return AgentConfig(
            id="test-agent",
            name="Test Agent",
            handles=["test"],
            url="http://test:8080",
            protocol=AgentProtocol.AG_UI,
            routing=AgentRoutingConfig(
                priority=1,
                threshold=0.8,
                examples=["test query"],
            ),
        )

    @pytest.fixture
    def chat_request(self):
        """Create a test chat request."""
        return ChatRequest(
            thread_id="test-thread",
            messages=[UserMessage(id="msg-1", role="user", content="Test message")],
        )

    @pytest.fixture
    def retry_config(self):
        """Create a test retry configuration with short delays."""
        return RetryConfig(
            max_attempts=3,
            base_delay_ms=10,  # Short delays for tests
            max_delay_ms=50,
        )

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self, agent_config, chat_request, retry_config):
        """Test successful request on first attempt."""
        proxy = AgentProxy(retry_config=retry_config)

        # Mock the AGUI client
        mock_events = [
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START, message_id="msg1", role="assistant"
            ),
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT, message_id="msg1", delta="Hello"
            ),
            TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id="msg1"),
        ]

        async def mock_send_message(*args, **kwargs):
            for event in mock_events:
                yield event

        with patch.object(proxy, "_ensure_clients", new_callable=AsyncMock):
            proxy._agui_client = MagicMock()
            proxy._agui_client.send_message = mock_send_message

            events = []
            async for event in proxy.forward_request(agent_config, chat_request):
                events.append(event)

        assert len(events) == 3
        assert events[0].type == EventType.TEXT_MESSAGE_START
        assert events[1].type == EventType.TEXT_MESSAGE_CONTENT
        assert events[2].type == EventType.TEXT_MESSAGE_END

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self, agent_config, chat_request, retry_config):
        """Test retry on connection error."""
        proxy = AgentProxy(retry_config=retry_config)

        call_count = 0

        async def mock_send_message_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise AGUIClientError("Connection timeout")
            # Success on third attempt
            yield TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START, message_id="msg1", role="assistant"
            )
            yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id="msg1")

        with patch.object(proxy, "_ensure_clients", new_callable=AsyncMock):
            proxy._agui_client = MagicMock()
            proxy._agui_client.send_message = mock_send_message_with_retry

            events = []
            async for event in proxy.forward_request(agent_config, chat_request):
                events.append(event)

        assert call_count == 3  # Two failures, one success
        assert len(events) == 2
        assert events[0].type == EventType.TEXT_MESSAGE_START

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, agent_config, chat_request, retry_config):
        """Test error raised after max retries exhausted."""
        proxy = AgentProxy(retry_config=retry_config)

        call_count = 0

        async def mock_send_message_always_fails(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise AGUIClientError("Connection timeout")
            yield  # Make this a generator

        with patch.object(proxy, "_ensure_clients", new_callable=AsyncMock):
            proxy._agui_client = MagicMock()
            proxy._agui_client.send_message = mock_send_message_always_fails

            events = []
            with pytest.raises(AgentProxyError) as exc_info:
                async for event in proxy.forward_request(agent_config, chat_request):
                    events.append(event)

        assert call_count == 3  # All attempts exhausted
        assert exc_info.value.attempts == 3
        assert "Test Agent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self, agent_config, chat_request, retry_config):
        """Test no retry on 4xx client errors."""
        proxy = AgentProxy(retry_config=retry_config)

        call_count = 0

        async def mock_send_message_client_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise AGUIClientError("HTTP 400 Bad Request", status_code=400)
            yield  # Make this a generator

        with patch.object(proxy, "_ensure_clients", new_callable=AsyncMock):
            proxy._agui_client = MagicMock()
            proxy._agui_client.send_message = mock_send_message_client_error

            events = []
            with pytest.raises(AgentProxyError):
                async for event in proxy.forward_request(agent_config, chat_request):
                    events.append(event)

        # Should not retry on 400 error
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self, agent_config, chat_request, retry_config):
        """Test retry on 5xx server errors."""
        proxy = AgentProxy(retry_config=retry_config)

        call_count = 0

        async def mock_send_message_server_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise AGUIClientError("HTTP 503 Service Unavailable", status_code=503)
            yield TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START, message_id="msg1", role="assistant"
            )
            yield TextMessageEndEvent(type=EventType.TEXT_MESSAGE_END, message_id="msg1")

        with patch.object(proxy, "_ensure_clients", new_callable=AsyncMock):
            proxy._agui_client = MagicMock()
            proxy._agui_client.send_message = mock_send_message_server_error

            events = []
            async for event in proxy.forward_request(agent_config, chat_request):
                events.append(event)

        assert call_count == 2  # One retry, then success
        assert len(events) == 2


class TestEndpointFallback:
    """Test endpoint fallback behavior."""

    @pytest.mark.asyncio
    async def test_yield_fallback_context(self):
        """Test _yield_fallback_context generates proper events."""
        from router.agui.endpoint import _yield_fallback_context

        events = []
        async for event in _yield_fallback_context("Agent 'metrics' failed. "):
            events.append(event)

        assert len(events) == 3

        # First event: TextMessageStartEvent
        assert events[0].type == EventType.TEXT_MESSAGE_START
        assert events[0].role == "assistant"

        # Second event: TextMessageContentEvent with fallback message
        assert events[1].type == EventType.TEXT_MESSAGE_CONTENT
        assert "Notice:" in events[1].delta
        assert "Agent 'metrics' failed" in events[1].delta
        assert "Routing to general assistant" in events[1].delta

        # Third event: TextMessageEndEvent
        assert events[2].type == EventType.TEXT_MESSAGE_END
