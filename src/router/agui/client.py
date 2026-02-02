"""AG-UI client for calling AG-UI protocol agents.

This module provides an async HTTP client that sends requests to AG-UI agents
and streams SSE responses back to the caller.
"""

import logging
import uuid
from collections.abc import AsyncGenerator

import httpx

from router.agui.models import (
    BaseEvent,
    ChatRequest,
    EventEncoder,
    RunAgentInput,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 60.0  # seconds
SSE_LINE_PREFIX = "data: "
SSE_EVENT_PREFIX = "event: "


class AGUIClientError(Exception):
    """Exception raised when AG-UI client operations fail."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class AGUIClient:
    """Async HTTP client for calling AG-UI protocol agents.

    This client sends chat requests to AG-UI agents and streams back SSE events.
    It handles the AG-UI protocol format and provides async generator iteration
    over response events.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize the AG-UI client.

        Args:
            http_client: Async HTTP client for making requests.
            timeout: Request timeout in seconds.
        """
        self._http_client = http_client
        self._timeout = timeout
        self._encoder = EventEncoder()

    async def send_message(
        self,
        url: str,
        request: ChatRequest,
        headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Send a message to an AG-UI agent and stream the response.

        Args:
            url: The agent's AG-UI endpoint URL.
            request: The chat request to send.
            headers: Optional headers to forward (Authorization, X-Request-ID, etc.)

        Yields:
            BaseEvent: AG-UI events from the agent response stream.

        Raises:
            AGUIClientError: If the request fails or the agent returns an error.
        """
        # Generate run_id for this request
        run_id = str(uuid.uuid4())

        # Build the full RunAgentInput
        agent_input = RunAgentInput(
            thread_id=request.thread_id,
            run_id=run_id,
            messages=request.messages,
            tools=request.tools,
            context=request.context,
            state=request.state,
            forwarded_props={},
        )

        # Prepare headers
        request_headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if headers:
            # Forward specified headers (Authorization, X-Request-ID, etc.)
            for key in ("Authorization", "X-Request-ID", "Content-Type"):
                if key in headers:
                    request_headers[key] = headers[key]

        logger.debug(
            "Sending AG-UI request to %s (thread_id=%s, run_id=%s)",
            url,
            request.thread_id,
            run_id,
        )

        try:
            async with self._http_client.stream(
                "POST",
                url,
                json=agent_input.model_dump(by_alias=True, exclude_none=True),
                headers=request_headers,
                timeout=self._timeout,
            ) as response:
                if response.status_code >= 400:
                    error_body = await response.aread()
                    error_msg = error_body.decode("utf-8", errors="replace")
                    logger.error(
                        "AG-UI agent returned error: status=%d, body=%s",
                        response.status_code,
                        error_msg[:200],
                    )
                    raise AGUIClientError(
                        f"Agent returned HTTP {response.status_code}: {error_msg[:200]}",
                        status_code=response.status_code,
                    )

                # Stream SSE events
                async for event in self._parse_sse_stream(response):
                    yield event

        except httpx.TimeoutException as e:
            logger.error("AG-UI request timed out: %s", e)
            raise AGUIClientError(f"Request timed out: {e}") from e
        except httpx.RequestError as e:
            logger.error("AG-UI request failed: %s", e)
            raise AGUIClientError(f"Request failed: {e}") from e

    async def _parse_sse_stream(
        self,
        response: httpx.Response,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Parse SSE events from an HTTP response stream.

        Args:
            response: The HTTP response with SSE content.

        Yields:
            BaseEvent: Parsed AG-UI events.
        """

        current_event_type: str | None = None
        current_data_lines: list[str] = []

        async for line in response.aiter_lines():
            line = line.strip()

            if not line:
                # Empty line signals end of event
                if current_data_lines:
                    data = "\n".join(current_data_lines)
                    event = self._parse_event(data, current_event_type)
                    if event:
                        yield event
                current_event_type = None
                current_data_lines = []
                continue

            if line.startswith(SSE_EVENT_PREFIX):
                current_event_type = line[len(SSE_EVENT_PREFIX) :].strip()
            elif line.startswith(SSE_LINE_PREFIX):
                current_data_lines.append(line[len(SSE_LINE_PREFIX) :])
            elif line.startswith(":"):
                # Comment, ignore
                continue

        # Handle any remaining data
        if current_data_lines:
            data = "\n".join(current_data_lines)
            event = self._parse_event(data, current_event_type)
            if event:
                yield event

    def _parse_event(self, data: str, event_type: str | None) -> BaseEvent | None:
        """Parse a single SSE event into an AG-UI event object.

        Args:
            data: The JSON data from the SSE event.
            event_type: The event type from the SSE event: header.

        Returns:
            Parsed BaseEvent or None if parsing fails.
        """
        import json

        if not data or data == "[DONE]":
            return None

        try:
            event_data = json.loads(data)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse SSE event JSON: %s", e)
            return None

        # Validate event has a type field
        if not event_data.get("type") and not event_type:
            logger.warning("SSE event missing type field")
            return None

        # Use event: header as fallback if type not in data
        if not event_data.get("type") and event_type:
            event_data["type"] = event_type

        try:
            return BaseEvent(**event_data)
        except Exception as e:
            logger.warning("Failed to create AG-UI event: %s", e)
            return None
