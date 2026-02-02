"""Stream buffer for reassembling SSE message chunks.

This module provides the StreamBuffer class that accumulates streaming
message events and reassembles them into complete messages for audit
logging and debugging purposes.
"""

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from router.observability.models import BufferedMessage, StreamFrame

logger = logging.getLogger(__name__)

# Event type constants
EVENT_TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
EVENT_TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
EVENT_TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
EVENT_RUN_STARTED = "RUN_STARTED"
EVENT_RUN_FINISHED = "RUN_FINISHED"
EVENT_RUN_ERROR = "RUN_ERROR"


class StreamBuffer:
    """Buffer for accumulating and reassembling streaming message events.

    This class tracks SSE events as they flow through the router, building
    complete messages from the stream of TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT,
    and TEXT_MESSAGE_END events.

    The buffer operates passthrough-style: events are yielded immediately while
    also being accumulated for audit logging purposes.

    Usage:
        buffer = StreamBuffer(thread_id="t1", run_id="r1", request_id="req1")
        async for event in buffer.process_stream(event_generator):
            # event is yielded immediately
            pass

        # After stream completes, get the buffered message
        message = buffer.get_message()
        if message and message.complete:
            audit_logger.log_message_complete(message)
    """

    def __init__(
        self,
        thread_id: str,
        run_id: str,
        request_id: str,
        max_content_size: int = 1_000_000,  # 1MB default
    ) -> None:
        """Initialize the stream buffer.

        Args:
            thread_id: The conversation thread ID.
            run_id: The run ID for this request.
            request_id: The request ID for correlation.
            max_content_size: Maximum accumulated content size in bytes.
        """
        self._thread_id = thread_id
        self._run_id = run_id
        self._request_id = request_id
        self._max_content_size = max_content_size
        self._message: BufferedMessage | None = None
        self._sequence_num = 0
        self._content_size = 0

    @property
    def message(self) -> BufferedMessage | None:
        """Get the current buffered message."""
        return self._message

    @property
    def is_complete(self) -> bool:
        """Check if the buffered message is complete."""
        return self._message is not None and self._message.complete

    def get_message(self) -> BufferedMessage | None:
        """Get the buffered message.

        Returns:
            The buffered message, or None if no message was started.
        """
        return self._message

    async def process_stream(
        self,
        event_stream: AsyncGenerator[dict[str, str], None],
    ) -> AsyncGenerator[dict[str, str], None]:
        """Process an SSE event stream, buffering while yielding passthrough.

        This method wraps an event generator, accumulating message content
        while immediately yielding each event to maintain streaming behavior.

        Args:
            event_stream: The source event generator yielding SSE dict events.

        Yields:
            dict: Each event from the source stream (passthrough).
        """
        async for event in event_stream:
            self._process_event(event)
            yield event

    def _process_event(self, event: dict[str, str]) -> None:
        """Process a single SSE event and update the buffer.

        Args:
            event: The SSE event dict with 'event' and 'data' keys.
        """
        event_type = event.get("event", "")
        data_str = event.get("data", "")

        # Parse the data JSON
        try:
            import json

            data = json.loads(data_str) if data_str else {}
        except (json.JSONDecodeError, TypeError):
            data = {}

        # Create a frame for this event
        frame = StreamFrame(
            event_type=event_type,
            data=data,
            timestamp=time.time(),
            sequence_num=self._sequence_num,
        )
        self._sequence_num += 1

        # Handle message lifecycle events
        if event_type == EVENT_TEXT_MESSAGE_START:
            self._handle_message_start(frame, data)
        elif event_type == EVENT_TEXT_MESSAGE_CONTENT:
            self._handle_message_content(frame, data)
        elif event_type == EVENT_TEXT_MESSAGE_END:
            self._handle_message_end(frame)
        elif event_type == EVENT_RUN_ERROR:
            self._handle_error(frame, data)

    def _handle_message_start(self, frame: StreamFrame, data: dict[str, Any]) -> None:
        """Handle TEXT_MESSAGE_START event.

        Args:
            frame: The stream frame.
            data: The parsed event data.
        """
        message_id = data.get("messageId") or data.get("message_id", "")

        self._message = BufferedMessage(
            message_id=message_id,
            thread_id=self._thread_id,
            run_id=self._run_id,
            frames=[frame],
            start_time=frame.timestamp,
        )
        self._content_size = 0

        logger.debug(
            "Message started (request_id=%s, message_id=%s)",
            self._request_id,
            message_id,
        )

    def _handle_message_content(self, frame: StreamFrame, data: dict[str, Any]) -> None:
        """Handle TEXT_MESSAGE_CONTENT event.

        Args:
            frame: The stream frame.
            data: The parsed event data.
        """
        if self._message is None:
            # Content without start - create message
            message_id = data.get("messageId") or data.get("message_id", "")
            self._message = BufferedMessage(
                message_id=message_id,
                thread_id=self._thread_id,
                run_id=self._run_id,
                frames=[frame],
                start_time=frame.timestamp,
            )

        self._message.frames.append(frame)

        # Accumulate content (with size limit)
        delta = data.get("delta", "")
        if delta and self._content_size < self._max_content_size:
            remaining = self._max_content_size - self._content_size
            truncated_delta = delta[:remaining]
            self._message.accumulated_content += truncated_delta
            self._content_size += len(truncated_delta)

            if len(delta) > remaining:
                logger.warning(
                    "Content truncated at max size (request_id=%s, max=%d)",
                    self._request_id,
                    self._max_content_size,
                )

    def _handle_message_end(self, frame: StreamFrame) -> None:
        """Handle TEXT_MESSAGE_END event.

        Args:
            frame: The stream frame.
        """
        if self._message is not None:
            self._message.frames.append(frame)
            self._message.complete = True
            self._message.completion_time = frame.timestamp

            duration_ms = 0.0
            if self._message.start_time:
                duration_ms = (frame.timestamp - self._message.start_time) * 1000

            logger.debug(
                "Message complete (request_id=%s, message_id=%s, "
                "content_length=%d, duration_ms=%.2f)",
                self._request_id,
                self._message.message_id,
                len(self._message.accumulated_content),
                duration_ms,
            )

    def _handle_error(self, frame: StreamFrame, data: dict[str, Any]) -> None:
        """Handle RUN_ERROR event.

        Args:
            frame: The stream frame.
            data: The parsed event data.
        """
        if self._message is not None:
            self._message.frames.append(frame)
            self._message.complete = True
            self._message.completion_time = frame.timestamp

        error_msg = data.get("message", "Unknown error")
        logger.debug(
            "Stream error (request_id=%s, error=%s)",
            self._request_id,
            error_msg[:100],
        )

    def get_stats(self) -> dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer statistics.
        """
        stats = {
            "request_id": self._request_id,
            "thread_id": self._thread_id,
            "run_id": self._run_id,
            "total_frames": self._sequence_num,
            "content_size": self._content_size,
            "has_message": self._message is not None,
            "is_complete": self.is_complete,
        }

        if self._message:
            stats["message_id"] = self._message.message_id
            stats["frame_count"] = len(self._message.frames)

            if self._message.start_time and self._message.completion_time:
                stats["duration_ms"] = (
                    self._message.completion_time - self._message.start_time
                ) * 1000

        return stats
