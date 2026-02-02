"""Protocol translator between AG-UI and A2A formats.

This module provides bidirectional translation between AG-UI and A2A protocol
message formats, enabling the router to accept AG-UI requests and forward them
to A2A agents, then translate the responses back to AG-UI events.
"""

import logging
import uuid
from collections.abc import AsyncGenerator

from a2a.types import (
    Role as A2ARole,
)
from a2a.types import (
    SendStreamingMessageResponse,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)

from router.agui.models import (
    BaseEvent,
    ChatRequest,
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)

logger = logging.getLogger(__name__)


class A2ATranslator:
    """Translator between AG-UI and A2A protocol formats.

    This class handles the conversion of requests and responses between the two
    protocols, allowing the router to bridge AG-UI clients with A2A agents.
    """

    @staticmethod
    def extract_last_user_message(request: ChatRequest) -> str:
        """Extract the last user message content from an AG-UI request.

        Args:
            request: The AG-UI chat request.

        Returns:
            The text content of the last user message.

        Raises:
            ValueError: If no user message is found.
        """
        for message in reversed(request.messages):
            if message.role == "user":
                # Handle different content types
                content = message.content
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    # Extract text from input content items
                    texts = []
                    for item in content:
                        if hasattr(item, "text"):
                            texts.append(item.text)
                    if texts:
                        return " ".join(texts)
        raise ValueError("No user message found in request")

    @staticmethod
    def thread_id_to_context_id(thread_id: str) -> str:
        """Convert AG-UI threadId to A2A contextId.

        The router uses threadId as the canonical session identifier.
        For A2A agents, this is translated to contextId.

        Args:
            thread_id: The AG-UI thread identifier.

        Returns:
            The A2A context identifier (same value).
        """
        return thread_id

    @staticmethod
    def context_id_to_thread_id(context_id: str) -> str:
        """Convert A2A contextId back to AG-UI threadId.

        Args:
            context_id: The A2A context identifier.

        Returns:
            The AG-UI thread identifier (same value).
        """
        return context_id

    def translate_a2a_stream_to_agui(
        self,
        stream: AsyncGenerator[SendStreamingMessageResponse, None],
        thread_id: str,
        run_id: str,
        display_name: str | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Translate an A2A streaming response to AG-UI events.

        This method wraps an A2A response stream and yields AG-UI events
        that can be streamed back to AG-UI clients.

        Args:
            stream: The A2A streaming response generator.
            thread_id: The AG-UI thread ID for the conversation.
            run_id: The run ID for this request.
            display_name: Optional display name to include in RUN_STARTED event.

        Yields:
            BaseEvent: AG-UI events translated from A2A responses.
        """
        return self._translate_stream(stream, thread_id, run_id, display_name)

    async def _translate_stream(
        self,
        stream: AsyncGenerator[SendStreamingMessageResponse, None],
        thread_id: str,
        run_id: str,
        display_name: str | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Internal implementation of stream translation.

        Args:
            stream: The A2A streaming response generator.
            thread_id: The AG-UI thread ID.
            run_id: The run ID.
            display_name: Optional display name for the agent.

        Yields:
            BaseEvent: AG-UI events.
        """
        message_id = str(uuid.uuid4())
        message_started = False
        accumulated_text = ""

        try:
            # Emit run started event
            run_started_event = RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=thread_id,
                run_id=run_id,
            )
            # Inject displayName if provided
            if display_name and hasattr(run_started_event, "model_dump"):
                event_data = run_started_event.model_dump(by_alias=True, exclude_none=True)
                event_data["displayName"] = display_name
                run_started_event = RunStartedEvent(**event_data)
            yield run_started_event

            async for response in stream:
                events = self._translate_response(
                    response,
                    message_id,
                    message_started,
                    accumulated_text,
                )

                for event, new_started, new_text in events:
                    message_started = new_started
                    accumulated_text = new_text
                    if event:
                        yield event

            # Ensure message is ended if started
            if message_started:
                yield TextMessageEndEvent(
                    type=EventType.TEXT_MESSAGE_END,
                    message_id=message_id,
                )

            # Emit run finished event
            yield RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=thread_id,
                run_id=run_id,
            )

        except Exception as e:
            logger.error("Error translating A2A stream: %s", e)
            yield RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=f"Error processing agent response: {e}",
            )

    def _translate_response(
        self,
        response: SendStreamingMessageResponse,
        message_id: str,
        message_started: bool,
        accumulated_text: str,
    ) -> list[tuple[BaseEvent | None, bool, str]]:
        """Translate a single A2A response to AG-UI events.

        Args:
            response: The A2A streaming response.
            message_id: The AG-UI message ID.
            message_started: Whether the message has been started.
            accumulated_text: Text accumulated so far.

        Returns:
            List of (event, new_message_started, new_accumulated_text) tuples.
        """
        results: list[tuple[BaseEvent | None, bool, str]] = []
        wrapper = response.root

        # Extract the actual event from the wrapper
        # SendStreamingMessageSuccessResponse has a .result field with the actual event
        if isinstance(wrapper, SendStreamingMessageSuccessResponse):
            event = wrapper.result
        else:
            # Handle error responses or other types by skipping
            logger.warning("Unexpected A2A response type: %s", type(wrapper).__name__)
            return [(None, message_started, accumulated_text)]

        if isinstance(event, Task):
            # Full task response - extract agent messages
            # Task events often contain the complete message in history,
            # which may duplicate content already streamed via status updates.
            # Only emit if there's genuinely new content.
            text = self._extract_task_text(event)
            if text:
                # Skip if text is a duplicate of what we've already accumulated
                # This handles the case where final Task contains the full message
                if not self._is_duplicate_content(text, accumulated_text):
                    results.extend(
                        self._emit_text_events(text, message_id, message_started, accumulated_text)
                    )
                    accumulated_text += text
                    message_started = True

        elif isinstance(event, TaskStatusUpdateEvent):
            # Status update with possible message
            text = self._extract_status_text(event)
            if text:
                results.extend(
                    self._emit_text_events(text, message_id, message_started, accumulated_text)
                )
                accumulated_text += text
                message_started = True

        elif isinstance(event, TaskArtifactUpdateEvent):
            # Artifact update with possible text
            text = self._extract_artifact_text(event)
            if text:
                results.extend(
                    self._emit_text_events(text, message_id, message_started, accumulated_text)
                )
                accumulated_text += text
                message_started = True

        # If no events were added, return current state
        if not results:
            results.append((None, message_started, accumulated_text))

        return results

    def _emit_text_events(
        self,
        text: str,
        message_id: str,
        message_started: bool,
        accumulated_text: str,
    ) -> list[tuple[BaseEvent, bool, str]]:
        """Emit text message events for new content.

        Args:
            text: The new text content.
            message_id: The AG-UI message ID.
            message_started: Whether the message has been started.
            accumulated_text: Text accumulated so far.

        Returns:
            List of (event, message_started, accumulated_text) tuples.
        """
        results: list[tuple[BaseEvent, bool, str]] = []

        # Start message if needed
        if not message_started:
            results.append(
                (
                    TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=message_id,
                        role="assistant",
                    ),
                    True,
                    accumulated_text,
                )
            )

        # Emit content delta (only new content)
        # If text starts with accumulated, it's cumulative - extract the new part
        # Otherwise, text is a delta that should be appended
        if text.startswith(accumulated_text):
            new_content = text[len(accumulated_text):]
            new_accumulated = text
        else:
            new_content = text
            new_accumulated = accumulated_text + text

        if new_content:
            results.append(
                (
                    TextMessageContentEvent(
                        type=EventType.TEXT_MESSAGE_CONTENT,
                        message_id=message_id,
                        delta=new_content,
                    ),
                    True,
                    new_accumulated,
                )
            )

        return results

    @staticmethod
    def _extract_task_text(task: Task) -> str | None:
        """Extract text from a Task object.

        Args:
            task: The A2A Task object.

        Returns:
            Extracted text or None.
        """
        if task.history:
            for msg in task.history:
                if msg.role == A2ARole.agent:
                    for part in msg.parts:
                        if isinstance(part.root, TextPart):
                            return part.root.text
        return None

    @staticmethod
    def _extract_status_text(event: TaskStatusUpdateEvent) -> str | None:
        """Extract text from a TaskStatusUpdateEvent.

        Args:
            event: The status update event.

        Returns:
            Extracted text or None.
        """
        if event.status and event.status.message:
            for part in event.status.message.parts:
                if isinstance(part.root, TextPart):
                    return part.root.text
        return None

    @staticmethod
    def _extract_artifact_text(event: TaskArtifactUpdateEvent) -> str | None:
        """Extract text from a TaskArtifactUpdateEvent.

        Args:
            event: The artifact update event.

        Returns:
            Extracted text or None.
        """
        if event.artifact and event.artifact.parts:
            for part in event.artifact.parts:
                if isinstance(part.root, TextPart):
                    return part.root.text
        return None

    @staticmethod
    def _is_duplicate_content(text: str, accumulated_text: str) -> bool:
        """Check if text is duplicate content that shouldn't be emitted.

        This detects when a final Task event contains the complete message
        that was already streamed via status update events.

        Args:
            text: The text from the current event.
            accumulated_text: The text accumulated from previous events.

        Returns:
            True if the text is a duplicate and should be skipped.
        """
        if not accumulated_text:
            # Nothing accumulated yet, so this is new content
            return False

        # Exact match - definitely a duplicate
        if text == accumulated_text:
            return True

        # Text is contained within accumulated - duplicate
        if text in accumulated_text:
            return True

        # Accumulated text starts with this text - it's a subset of what we have
        if accumulated_text.startswith(text):
            return True

        # Text starts with accumulated - this would be new content appended
        # In this case, the new content is text[len(accumulated_text):]
        # Let _emit_text_events handle this normally
        if text.startswith(accumulated_text):
            return False

        # For other cases, check if they're similar enough to be duplicates
        # This handles cases where there are minor whitespace differences
        text_normalized = " ".join(text.split())
        accumulated_normalized = " ".join(accumulated_text.split())

        if text_normalized == accumulated_normalized:
            return True

        # If the text is very similar in length to accumulated (within 10%),
        # and accumulated is substantial, it's likely a duplicate with minor formatting
        if len(accumulated_text) > 50:
            length_ratio = len(text) / len(accumulated_text)
            if 0.9 <= length_ratio <= 1.1:
                # Check character overlap
                common_prefix = 0
                for a, b in zip(text, accumulated_text, strict=False):
                    if a == b:
                        common_prefix += 1
                    else:
                        break
                # If >80% of chars match at the start, it's likely a duplicate
                if common_prefix > len(accumulated_text) * 0.8:
                    return True

        return False
