"""Agent proxy for forwarding requests to agents.

This module provides the AgentProxy class that orchestrates forwarding requests
to the appropriate agent based on its protocol (AG-UI or A2A) and handles
response streaming back to the caller.
"""

import logging
import uuid
from collections.abc import AsyncGenerator

import httpx

from router.a2a.client import A2AClientError, A2AClientWrapper
from router.a2a.translator import A2ATranslator
from router.agents.retry import RetryConfig, is_retryable_error
from router.agui.client import AGUIClient, AGUIClientError
from router.agui.models import (
    BaseEvent,
    ChatRequest,
    EventType,
    RunErrorEvent,
    inject_display_name,
)
from router.config import AgentConfig, AgentProtocol

logger = logging.getLogger(__name__)


class AgentProxyError(Exception):
    """Exception raised when agent proxy operations fail."""

    def __init__(
        self,
        message: str,
        agent_id: str | None = None,
        agent_name: str | None = None,
        attempts: int = 1,
        is_retryable: bool = False,
    ):
        super().__init__(message)
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.attempts = attempts
        self.is_retryable = is_retryable


class AgentProxy:
    """Proxy for forwarding requests to agents.

    This class handles the complexity of routing requests to different agents
    based on their protocol (AG-UI or A2A). It manages HTTP clients, protocol
    translation, response streaming, and retry logic with exponential backoff.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        retry_config: RetryConfig | None = None,
    ):
        """Initialize the agent proxy.

        Args:
            http_client: Optional shared HTTP client. If not provided,
                a new client will be created.
            timeout: Request timeout in seconds.
            retry_config: Optional retry configuration. If not provided,
                defaults to 3 attempts with 500ms base delay.
        """
        self._http_client = http_client
        self._owns_client = http_client is None
        self._timeout = timeout
        self._retry_config = retry_config or RetryConfig()
        self._agui_client: AGUIClient | None = None
        self._a2a_client: A2AClientWrapper | None = None
        self._translator = A2ATranslator()

    async def _ensure_clients(self) -> None:
        """Ensure HTTP clients are initialized."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._timeout)
            self._owns_client = True

        if self._agui_client is None:
            self._agui_client = AGUIClient(self._http_client, timeout=self._timeout)

        if self._a2a_client is None:
            self._a2a_client = A2AClientWrapper(self._http_client, timeout=self._timeout)

    async def close(self) -> None:
        """Close the proxy and release resources."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def forward_request(
        self,
        agent: AgentConfig,
        request: ChatRequest,
        headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Forward a request to an agent and stream the response.

        This method determines the agent's protocol and routes the request
        appropriately. For AG-UI agents, it forwards the request directly.
        For A2A agents, it translates the request and response.

        Retry behavior:
        - Retries on connection errors, timeouts, and 5xx errors
        - Uses exponential backoff between retries
        - Does not retry on 4xx client errors (except 429)
        - Retries only before events have started streaming

        Args:
            agent: The agent configuration.
            request: The AG-UI chat request to forward.
            headers: Optional headers to forward (Authorization, X-Request-ID).

        Yields:
            BaseEvent: AG-UI events from the agent response.

        Raises:
            AgentProxyError: If the request fails after all retries.
        """
        await self._ensure_clients()

        logger.info(
            "Forwarding request to agent %s (%s) via %s",
            agent.name,
            agent.id,
            agent.protocol.value,
        )

        last_error: Exception | None = None
        attempts = 0

        for attempt in range(self._retry_config.max_attempts):
            attempts = attempt + 1

            try:
                # Wait before retry (no wait for first attempt)
                if attempt > 0:
                    await self._retry_config.wait_before_retry(attempt)
                    logger.info(
                        "Retrying request to agent %s (attempt %d/%d)",
                        agent.name,
                        attempts,
                        self._retry_config.max_attempts,
                    )

                # Try to forward the request
                async for event in self._forward_with_protocol(agent, request, headers):
                    yield event

                # If we get here, the request succeeded
                return

            except (AGUIClientError, A2AClientError) as e:
                last_error = e
                retryable = is_retryable_error(e)

                if not retryable:
                    logger.error("Agent request failed with non-retryable error: %s", e)
                    break

                if attempt < self._retry_config.max_attempts - 1:
                    logger.warning(
                        "Agent request failed (attempt %d/%d): %s",
                        attempts,
                        self._retry_config.max_attempts,
                        e,
                    )
                else:
                    logger.error(
                        "Agent request failed after %d attempts: %s",
                        attempts,
                        e,
                    )

        # All retries exhausted or non-retryable error
        error_msg = f"Agent {agent.name} failed after {attempts} attempt(s): {last_error}"
        yield RunErrorEvent(
            type=EventType.RUN_ERROR,
            message=error_msg,
        )
        raise AgentProxyError(
            error_msg,
            agent_id=agent.id,
            agent_name=agent.name,
            attempts=attempts,
            is_retryable=is_retryable_error(last_error) if last_error else False,
        )

    async def _forward_with_protocol(
        self,
        agent: AgentConfig,
        request: ChatRequest,
        headers: dict[str, str] | None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Forward request based on agent protocol.

        Args:
            agent: The agent configuration.
            request: The chat request.
            headers: Optional headers to forward.

        Yields:
            BaseEvent: AG-UI events from the agent.
        """
        if agent.protocol == AgentProtocol.AG_UI:
            async for event in self._forward_agui(agent, request, headers):
                yield event
        else:  # A2A protocol
            async for event in self._forward_a2a(agent, request, headers):
                yield event

    async def _forward_agui(
        self,
        agent: AgentConfig,
        request: ChatRequest,
        headers: dict[str, str] | None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Forward a request to an AG-UI protocol agent.

        Args:
            agent: The agent configuration.
            request: The chat request.
            headers: Optional headers to forward.

        Yields:
            BaseEvent: AG-UI events from the agent, with displayName injected
            into RUN_STARTED events.
        """
        # AG-UI agents receive the request directly
        url = str(agent.url)
        async for event in self._agui_client.send_message(url, request, headers):
            # Inject displayName into RUN_STARTED events
            yield inject_display_name(event, agent.name)

    async def _forward_a2a(
        self,
        agent: AgentConfig,
        request: ChatRequest,
        headers: dict[str, str] | None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Forward a request to an A2A protocol agent.

        This method translates the AG-UI request to A2A format, sends it to
        the agent, and translates the A2A response back to AG-UI events.

        Args:
            agent: The agent configuration.
            request: The chat request.
            headers: Optional headers to forward.

        Yields:
            BaseEvent: AG-UI events translated from the A2A response, with
            displayName included in the RUN_STARTED event.
        """
        # Extract the user message content
        try:
            content = self._translator.extract_last_user_message(request)
        except ValueError as e:
            logger.error("Failed to extract user message: %s", e)
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=f"Invalid request: {e}",
            )
            return

        # Convert threadId to contextId
        context_id = self._translator.thread_id_to_context_id(request.thread_id)

        # Generate run_id for this request
        run_id = str(uuid.uuid4())

        # Send to A2A agent and translate response
        url = str(agent.url)
        a2a_stream = self._a2a_client.send_message_streaming(url, content, context_id, headers)

        # Pass display_name to translator which will include it in RUN_STARTED event
        async for event in self._translator.translate_a2a_stream_to_agui(
            a2a_stream, request.thread_id, run_id, display_name=agent.name
        ):
            yield event
