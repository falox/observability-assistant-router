"""Tests for the A2A to AG-UI translator."""

import pytest
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    SendStreamingMessageResponse,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from router.a2a.translator import A2ATranslator
from router.agui.models import AssistantMessage, ChatRequest, EventType, UserMessage


class TestIsDuplicateContent:
    """Tests for the _is_duplicate_content method.

    This method detects when a final Task event contains duplicate content
    that was already streamed via TaskStatusUpdateEvent events, preventing
    the message from being sent twice to the client.
    """

    def test_empty_accumulated_is_not_duplicate(self):
        """Text is not duplicate when nothing has been accumulated yet."""
        assert A2ATranslator._is_duplicate_content("Hello", "") is False

    def test_exact_match_is_duplicate(self):
        """Exact text match is a duplicate."""
        text = "Hello, world!"
        assert A2ATranslator._is_duplicate_content(text, text) is True

    def test_text_contained_in_accumulated_is_duplicate(self):
        """Text that is contained within accumulated text is a duplicate."""
        accumulated = "Hello, world! How are you?"
        text = "Hello, world!"
        assert A2ATranslator._is_duplicate_content(text, accumulated) is True

    def test_accumulated_starts_with_text_is_duplicate(self):
        """Text that accumulated starts with is a duplicate (subset)."""
        accumulated = "Hello, world!"
        text = "Hello"
        assert A2ATranslator._is_duplicate_content(text, accumulated) is True

    def test_text_extends_accumulated_is_not_duplicate(self):
        """Text that extends accumulated (new content appended) is not duplicate."""
        accumulated = "Hello"
        text = "Hello, world!"  # New content: ", world!"
        assert A2ATranslator._is_duplicate_content(text, accumulated) is False

    def test_completely_different_text_is_not_duplicate(self):
        """Completely different text is not a duplicate."""
        accumulated = "Hello"
        text = "Goodbye"
        assert A2ATranslator._is_duplicate_content(text, accumulated) is False

    def test_whitespace_normalized_match_is_duplicate(self):
        """Text that matches after whitespace normalization is duplicate."""
        accumulated = "Hello   world"
        text = "Hello world"
        assert A2ATranslator._is_duplicate_content(text, accumulated) is True

    def test_multiline_whitespace_normalized_is_duplicate(self):
        """Text with different line breaks but same content is duplicate."""
        accumulated = "Hello\n\nworld"
        text = "Hello world"
        assert A2ATranslator._is_duplicate_content(text, accumulated) is True

    def test_similar_length_high_overlap_is_duplicate(self):
        """Text of similar length with high character overlap is duplicate."""
        # This tests the heuristic for minor formatting differences
        accumulated = "This is a test message with some content that is quite long."
        # Same text with minor difference at the end
        text = "This is a test message with some content that is quite long!"
        assert A2ATranslator._is_duplicate_content(text, accumulated) is True

    def test_similar_length_low_overlap_is_not_duplicate(self):
        """Text of similar length but different content is not duplicate."""
        accumulated = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        text = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
        assert A2ATranslator._is_duplicate_content(text, accumulated) is False

    def test_short_accumulated_not_affected_by_heuristic(self):
        """Short accumulated text doesn't trigger length-based heuristic."""
        accumulated = "Hi"
        text = "Ho"  # Same length, different content
        assert A2ATranslator._is_duplicate_content(text, accumulated) is False


class TestA2ATranslator:
    """Tests for the A2ATranslator class."""

    @pytest.fixture
    def translator(self):
        """Create a translator instance."""
        return A2ATranslator()

    def test_extract_last_user_message_string_content(self, translator):
        """Test extracting user message with string content."""
        request = ChatRequest(
            thread_id="thread-123",
            messages=[
                UserMessage(id="msg-1", role="user", content="Hello, agent!"),
            ],
        )
        result = translator.extract_last_user_message(request)
        assert result == "Hello, agent!"

    def test_extract_last_user_message_multiple_messages(self, translator):
        """Test extracting the last user message from multiple messages."""
        request = ChatRequest(
            thread_id="thread-123",
            messages=[
                UserMessage(id="msg-1", role="user", content="First message"),
                AssistantMessage(id="msg-2", role="assistant", content="Response"),
                UserMessage(id="msg-3", role="user", content="Second message"),
            ],
        )
        result = translator.extract_last_user_message(request)
        assert result == "Second message"

    def test_extract_last_user_message_no_user_message(self, translator):
        """Test error when no user message exists."""
        request = ChatRequest(
            thread_id="thread-123",
            messages=[
                AssistantMessage(id="msg-1", role="assistant", content="Hello"),
            ],
        )
        with pytest.raises(ValueError, match="No user message found"):
            translator.extract_last_user_message(request)

    def test_thread_id_to_context_id(self, translator):
        """Test threadId to contextId conversion."""
        thread_id = "thread-abc-123"
        result = translator.thread_id_to_context_id(thread_id)
        assert result == thread_id

    def test_context_id_to_thread_id(self, translator):
        """Test contextId to threadId conversion."""
        context_id = "context-xyz-789"
        result = translator.context_id_to_thread_id(context_id)
        assert result == context_id


class TestA2ATranslatorStreamTranslation:
    """Tests for stream translation functionality."""

    @pytest.fixture
    def translator(self):
        """Create a translator instance."""
        return A2ATranslator()

    @pytest.mark.asyncio
    async def test_translate_empty_stream(self, translator):
        """Test translating an empty stream."""

        async def empty_stream():
            return
            yield  # Make it a generator

        events = []
        async for event in translator.translate_a2a_stream_to_agui(
            empty_stream(), "thread-123", "run-456"
        ):
            events.append(event)

        # Should have at least run_started and run_finished
        assert len(events) >= 2
        assert events[0].type.value == "RUN_STARTED"
        assert events[-1].type.value == "RUN_FINISHED"

    @pytest.mark.asyncio
    async def test_translate_status_update_with_text(self, translator):
        """Test translating TaskStatusUpdateEvent with text message."""

        async def status_stream():
            # Simulate A2A response with status update containing text
            status_update = TaskStatusUpdateEvent(
                kind="status-update",
                task_id="task-123",
                context_id="thread-123",
                final=False,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        message_id="msg-1",
                        context_id="thread-123",
                        role=Role.agent,
                        parts=[Part(root=TextPart(text="Hello"))],
                    ),
                ),
            )
            response = SendStreamingMessageResponse(
                root=SendStreamingMessageSuccessResponse(
                    id="req-1",
                    result=status_update,
                )
            )
            yield response

        events = []
        async for event in translator.translate_a2a_stream_to_agui(
            status_stream(), "thread-123", "run-456"
        ):
            events.append(event)

        # Should have: RUN_STARTED, TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT,
        # TEXT_MESSAGE_END, RUN_FINISHED
        assert len(events) == 5
        assert events[0].type == EventType.RUN_STARTED
        assert events[1].type == EventType.TEXT_MESSAGE_START
        assert events[2].type == EventType.TEXT_MESSAGE_CONTENT
        assert events[2].delta == "Hello"
        assert events[3].type == EventType.TEXT_MESSAGE_END
        assert events[4].type == EventType.RUN_FINISHED

    @pytest.mark.asyncio
    async def test_translate_artifact_update_with_text(self, translator):
        """Test translating TaskArtifactUpdateEvent with text."""

        async def artifact_stream():
            # Simulate A2A response with artifact containing text
            artifact_update = TaskArtifactUpdateEvent(
                kind="artifact-update",
                task_id="task-123",
                context_id="thread-123",
                artifact=Artifact(
                    artifact_id="art-1",
                    parts=[Part(root=TextPart(text="Complete response"))],
                ),
                last_chunk=True,
            )
            response = SendStreamingMessageResponse(
                root=SendStreamingMessageSuccessResponse(
                    id="req-1",
                    result=artifact_update,
                )
            )
            yield response

        events = []
        async for event in translator.translate_a2a_stream_to_agui(
            artifact_stream(), "thread-123", "run-456"
        ):
            events.append(event)

        # Should have: RUN_STARTED, TEXT_MESSAGE_START, TEXT_MESSAGE_CONTENT,
        # TEXT_MESSAGE_END, RUN_FINISHED
        assert len(events) == 5
        assert events[0].type == EventType.RUN_STARTED
        assert events[1].type == EventType.TEXT_MESSAGE_START
        assert events[2].type == EventType.TEXT_MESSAGE_CONTENT
        assert events[2].delta == "Complete response"
        assert events[3].type == EventType.TEXT_MESSAGE_END
        assert events[4].type == EventType.RUN_FINISHED

    @pytest.mark.asyncio
    async def test_translate_streaming_text_chunks(self, translator):
        """Test translating multiple status updates with streaming text."""

        async def streaming_text():
            # Simulate streaming text in chunks
            chunks = ["Hello", " ", "world", "!"]
            for chunk in chunks:
                status_update = TaskStatusUpdateEvent(
                    kind="status-update",
                    task_id="task-123",
                    context_id="thread-123",
                    final=False,
                    status=TaskStatus(
                        state=TaskState.working,
                        message=Message(
                            message_id=f"msg-{chunk}",
                            context_id="thread-123",
                            role=Role.agent,
                            parts=[Part(root=TextPart(text=chunk))],
                        ),
                    ),
                )
                response = SendStreamingMessageResponse(
                    root=SendStreamingMessageSuccessResponse(
                        id="req-1",
                        result=status_update,
                    )
                )
                yield response

        events = []
        async for event in translator.translate_a2a_stream_to_agui(
            streaming_text(), "thread-123", "run-456"
        ):
            events.append(event)

        # Filter content events
        content_events = [e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
        assert len(content_events) == 4
        assert content_events[0].delta == "Hello"
        assert content_events[1].delta == " "
        assert content_events[2].delta == "world"
        assert content_events[3].delta == "!"

    @pytest.mark.asyncio
    async def test_translate_task_with_agent_message(self, translator):
        """Test translating Task object with agent message in history."""

        async def task_stream():
            # Simulate A2A Task response with history
            task = Task(
                id="task-123",
                context_id="thread-123",
                status=TaskStatus(state=TaskState.completed),
                history=[
                    Message(
                        message_id="msg-1",
                        context_id="thread-123",
                        role=Role.user,
                        parts=[Part(root=TextPart(text="Hello"))],
                    ),
                    Message(
                        message_id="msg-2",
                        context_id="thread-123",
                        role=Role.agent,
                        parts=[Part(root=TextPart(text="Hi there!"))],
                    ),
                ],
            )
            response = SendStreamingMessageResponse(
                root=SendStreamingMessageSuccessResponse(
                    id="req-1",
                    result=task,
                )
            )
            yield response

        events = []
        async for event in translator.translate_a2a_stream_to_agui(
            task_stream(), "thread-123", "run-456"
        ):
            events.append(event)

        # Should extract text from agent message in history
        content_events = [e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]
        assert len(content_events) == 1
        assert content_events[0].delta == "Hi there!"

    @pytest.mark.asyncio
    async def test_translate_status_without_message(self, translator):
        """Test translating status update without message (no text)."""

        async def status_stream():
            # Status update without message
            status_update = TaskStatusUpdateEvent(
                kind="status-update",
                task_id="task-123",
                context_id="thread-123",
                final=False,
                status=TaskStatus(state=TaskState.working),
            )
            response = SendStreamingMessageResponse(
                root=SendStreamingMessageSuccessResponse(
                    id="req-1",
                    result=status_update,
                )
            )
            yield response

        events = []
        async for event in translator.translate_a2a_stream_to_agui(
            status_stream(), "thread-123", "run-456"
        ):
            events.append(event)

        # Should only have RUN_STARTED and RUN_FINISHED (no text events)
        assert len(events) == 2
        assert events[0].type == EventType.RUN_STARTED
        assert events[1].type == EventType.RUN_FINISHED

    @pytest.mark.asyncio
    async def test_final_task_with_duplicate_content_is_skipped(self, translator):
        """Test that final Task event with duplicate content doesn't cause double message.

        This is a regression test for the bug where:
        1. Streaming events (TaskStatusUpdateEvent) send token-by-token deltas
        2. Final Task event contains the complete message in history
        3. The complete message was being emitted again, causing duplicate content

        The fix detects when the Task event contains already-streamed content
        and skips emitting it again.
        """

        async def streaming_then_task():
            # First: Stream text via status updates (simulating token streaming)
            chunks = ["Hello", "!", " How", " are", " you", "?"]
            for chunk in chunks:
                status_update = TaskStatusUpdateEvent(
                    kind="status-update",
                    task_id="task-123",
                    context_id="thread-123",
                    final=False,
                    status=TaskStatus(
                        state=TaskState.working,
                        message=Message(
                            message_id=f"msg-{chunk}",
                            context_id="thread-123",
                            role=Role.agent,
                            parts=[Part(root=TextPart(text=chunk))],
                        ),
                    ),
                )
                yield SendStreamingMessageResponse(
                    root=SendStreamingMessageSuccessResponse(
                        id="req-1",
                        result=status_update,
                    )
                )

            # Then: Final Task event with complete message in history (this should be skipped)
            complete_message = "Hello! How are you?"
            task = Task(
                id="task-123",
                context_id="thread-123",
                status=TaskStatus(state=TaskState.completed),
                history=[
                    Message(
                        message_id="msg-final",
                        context_id="thread-123",
                        role=Role.agent,
                        parts=[Part(root=TextPart(text=complete_message))],
                    ),
                ],
            )
            yield SendStreamingMessageResponse(
                root=SendStreamingMessageSuccessResponse(
                    id="req-2",
                    result=task,
                )
            )

        events = []
        async for event in translator.translate_a2a_stream_to_agui(
            streaming_then_task(), "thread-123", "run-456"
        ):
            events.append(event)

        # Filter content events
        content_events = [e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]

        # Should have 6 content events (one per chunk), NOT 7 (with duplicate final message)
        assert len(content_events) == 6, (
            f"Expected 6 content events (one per streaming chunk), got {len(content_events)}. "
            "The final Task event should not emit duplicate content."
        )

        # Verify the deltas are the original chunks
        deltas = [e.delta for e in content_events]
        assert deltas == ["Hello", "!", " How", " are", " you", "?"]

    @pytest.mark.asyncio
    async def test_task_with_new_content_is_emitted(self, translator):
        """Test that Task event with genuinely new content is emitted.

        When Task event contains content that extends what was streamed,
        the new portion should be emitted.
        """

        async def streaming_then_extended_task():
            # First: Stream partial text
            status_update = TaskStatusUpdateEvent(
                kind="status-update",
                task_id="task-123",
                context_id="thread-123",
                final=False,
                status=TaskStatus(
                    state=TaskState.working,
                    message=Message(
                        message_id="msg-1",
                        context_id="thread-123",
                        role=Role.agent,
                        parts=[Part(root=TextPart(text="Hello"))],
                    ),
                ),
            )
            yield SendStreamingMessageResponse(
                root=SendStreamingMessageSuccessResponse(
                    id="req-1",
                    result=status_update,
                )
            )

            # Then: Task with extended content (new content should be emitted)
            task = Task(
                id="task-123",
                context_id="thread-123",
                status=TaskStatus(state=TaskState.completed),
                history=[
                    Message(
                        message_id="msg-final",
                        context_id="thread-123",
                        role=Role.agent,
                        parts=[Part(root=TextPart(text="Hello, world!"))],
                    ),
                ],
            )
            yield SendStreamingMessageResponse(
                root=SendStreamingMessageSuccessResponse(
                    id="req-2",
                    result=task,
                )
            )

        events = []
        async for event in translator.translate_a2a_stream_to_agui(
            streaming_then_extended_task(), "thread-123", "run-456"
        ):
            events.append(event)

        # Filter content events
        content_events = [e for e in events if e.type == EventType.TEXT_MESSAGE_CONTENT]

        # Should have 2 content events: "Hello" from status update, ", world!" from Task
        assert len(content_events) == 2
        assert content_events[0].delta == "Hello"
        assert content_events[1].delta == ", world!"

    @pytest.mark.asyncio
    async def test_translate_includes_display_name_in_run_started(self, translator):
        """Test that displayName is included in RUN_STARTED event when provided."""

        async def empty_stream():
            return
            yield  # Make it a generator

        events = []
        async for event in translator.translate_a2a_stream_to_agui(
            empty_stream(), "thread-123", "run-456", display_name="Prometheus Expert"
        ):
            events.append(event)

        # First event should be RUN_STARTED with displayName
        assert events[0].type == EventType.RUN_STARTED
        event_data = events[0].model_dump(by_alias=True)
        assert event_data.get("displayName") == "Prometheus Expert"
        assert event_data.get("threadId") == "thread-123"
        assert event_data.get("runId") == "run-456"

    @pytest.mark.asyncio
    async def test_translate_no_display_name_when_not_provided(self, translator):
        """Test that displayName is not present when not provided."""

        async def empty_stream():
            return
            yield  # Make it a generator

        events = []
        async for event in translator.translate_a2a_stream_to_agui(
            empty_stream(), "thread-123", "run-456"
        ):
            events.append(event)

        # First event should be RUN_STARTED without displayName
        assert events[0].type == EventType.RUN_STARTED
        event_data = events[0].model_dump(by_alias=True)
        assert "displayName" not in event_data or event_data.get("displayName") is None
