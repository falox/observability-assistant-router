"""A2A client wrapper for calling A2A protocol agents.

This module wraps the a2a-sdk client to provide a simplified interface
for sending messages and streaming responses from A2A agents.
"""

import logging
import uuid
from collections.abc import AsyncGenerator

import httpx
from a2a.client import A2AClient
from a2a.types import (
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    Role,
    SendStreamingMessageRequest,
    SendStreamingMessageResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 60.0  # seconds


class A2AClientError(Exception):
    """Exception raised when A2A client operations fail."""

    def __init__(self, message: str, error_code: str | None = None):
        super().__init__(message)
        self.error_code = error_code


class A2AClientWrapper:
    """Wrapper around the A2A SDK client for simplified usage.

    This wrapper provides a higher-level interface for sending messages
    to A2A agents and streaming responses. It handles message construction,
    context ID management, and response parsing.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize the A2A client wrapper.

        Args:
            http_client: Async HTTP client for making requests.
            timeout: Request timeout in seconds.
        """
        self._http_client = http_client
        self._timeout = timeout

    async def send_message_streaming(
        self,
        url: str,
        content: str,
        context_id: str,
        headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[SendStreamingMessageResponse, None]:
        """Send a message to an A2A agent and stream the response.

        Args:
            url: The agent's A2A endpoint URL.
            content: The message content to send.
            context_id: The context ID for the conversation (maps to AG-UI threadId).
            headers: Optional headers to forward (Authorization, etc.)

        Yields:
            SendStreamingMessageResponse: Streaming response events from the agent.

        Raises:
            A2AClientError: If the request fails or the agent returns an error.
        """
        # Create the A2A client for this request
        client = A2AClient(
            httpx_client=self._http_client,
            url=url,
        )

        # Build the message
        message = Message(
            message_id=str(uuid.uuid4()),
            context_id=context_id,
            role=Role.user,
            parts=[Part(root=TextPart(text=content))],
        )

        # Build the request
        request = SendStreamingMessageRequest(
            id=str(uuid.uuid4()),
            params=MessageSendParams(
                message=message,
                configuration=MessageSendConfiguration(
                    blocking=False,
                    accepted_output_modes=["text", "text/plain"],
                ),
            ),
        )

        logger.debug(
            "Sending A2A streaming request to %s (context_id=%s)",
            url,
            context_id,
        )

        try:
            # Add custom headers by setting them on the context
            http_kwargs = {"timeout": self._timeout}
            if headers:
                http_kwargs["headers"] = {
                    k: v for k, v in headers.items() if k in ("Authorization", "X-Request-ID")
                }

            async for response in client.send_message_streaming(request, http_kwargs=http_kwargs):
                yield response

        except Exception as e:
            logger.error("A2A request failed: %s", e)
            raise A2AClientError(f"A2A request failed: {e}") from e

    @staticmethod
    def extract_text_from_response(response: SendStreamingMessageResponse) -> str | None:
        """Extract text content from an A2A streaming response.

        Args:
            response: The streaming response event.

        Returns:
            Extracted text content or None if not a text response.
        """
        # SendStreamingMessageResponse uses a root model
        event = response.root

        if isinstance(event, Task):
            # Task object - extract from history or artifacts
            if event.history:
                for msg in event.history:
                    if msg.role == Role.agent:
                        for part in msg.parts:
                            if isinstance(part.root, TextPart):
                                return part.root.text
            return None

        if isinstance(event, TaskStatusUpdateEvent):
            # Status update - check if there's a message
            if event.status and event.status.message:
                for part in event.status.message.parts:
                    if isinstance(part.root, TextPart):
                        return part.root.text
            return None

        if isinstance(event, TaskArtifactUpdateEvent):
            # Artifact update - check for text content
            if event.artifact and event.artifact.parts:
                for part in event.artifact.parts:
                    if isinstance(part.root, TextPart):
                        return part.root.text
            return None

        return None

    @staticmethod
    def is_final_response(response: SendStreamingMessageResponse) -> bool:
        """Check if a response event is the final one.

        Args:
            response: The streaming response event.

        Returns:
            True if this is the final event in the stream.
        """
        event = response.root

        if isinstance(event, Task):
            # Task is complete if status is terminal
            return event.status.state in ("completed", "failed", "cancelled")

        if isinstance(event, TaskStatusUpdateEvent):
            return getattr(event, "final", False)

        if isinstance(event, TaskArtifactUpdateEvent):
            return getattr(event, "last_chunk", False)

        return False
