"""AG-UI protocol models and type definitions.

This module re-exports types from ag-ui-protocol and defines additional models
needed for the router's AG-UI endpoint.
"""

from typing import Annotated

# Re-export all AG-UI core types
from ag_ui.core import (
    AssistantMessage,
    BaseEvent,
    Context,
    Event,
    EventType,
    Message,
    Role,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    State,
    SystemMessage,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    Tool,
    ToolCall,
    ToolCallArgsEvent,
    ToolCallChunkEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
    ToolMessage,
    UserMessage,
)
from ag_ui.encoder import AGUI_MEDIA_TYPE, EventEncoder
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    # Re-exported from ag-ui-protocol
    "AssistantMessage",
    "BaseEvent",
    "Context",
    "Event",
    "EventType",
    "Message",
    "Role",
    "RunAgentInput",
    "RunErrorEvent",
    "RunFinishedEvent",
    "RunStartedEvent",
    "State",
    "SystemMessage",
    "TextMessageContentEvent",
    "TextMessageEndEvent",
    "TextMessageStartEvent",
    "Tool",
    "ToolCall",
    "ToolCallArgsEvent",
    "ToolCallChunkEvent",
    "ToolCallEndEvent",
    "ToolCallResultEvent",
    "ToolCallStartEvent",
    "ToolMessage",
    "UserMessage",
    "AGUI_MEDIA_TYPE",
    "EventEncoder",
    # Local models
    "ChatRequest",
    "inject_display_name",
]


# Maximum message content length to prevent DoS
MAX_CONTENT_LENGTH = 10000


class ChatRequest(BaseModel):
    """Request model for the AG-UI chat endpoint.

    This is a simplified version of RunAgentInput that accepts the minimum
    required fields from clients. The router will populate additional fields
    (run_id, state, etc.) before forwarding to agents.

    Accepts both camelCase (AG-UI protocol standard) and snake_case formats.
    """

    model_config = ConfigDict(populate_by_name=True)

    thread_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=100,
            description="Unique identifier for the conversation thread",
            alias="threadId",
        ),
    ]
    messages: Annotated[
        list[Message],
        Field(
            min_length=1,
            max_length=100,
            description="Conversation messages",
        ),
    ]
    tools: Annotated[
        list[Tool],
        Field(
            default_factory=list,
            max_length=50,
            description="Available tools for the agent",
        ),
    ]
    context: Annotated[
        list[Context],
        Field(
            default_factory=list,
            max_length=20,
            description="Contextual information for the agent",
        ),
    ]
    state: Annotated[
        State | None,
        Field(
            default=None,
            description="Current agent state",
        ),
    ]


def inject_display_name(event: BaseEvent, display_name: str | None) -> BaseEvent:
    """Inject displayName into a RUN_STARTED event.

    This function checks if the event is a RUN_STARTED event and adds the
    displayName field if provided. Other event types are passed through unchanged.

    Args:
        event: The AG-UI event to potentially modify.
        display_name: The display name to inject (agent name).

    Returns:
        The event, potentially with displayName added.
    """
    if display_name is None:
        return event

    # Check if this is a RUN_STARTED event
    event_type = getattr(event, "type", None)
    if event_type is None:
        return event

    # Get the event type value for comparison
    type_value = event_type.value if hasattr(event_type, "value") else str(event_type)
    if type_value != EventType.RUN_STARTED.value:
        return event

    # Create a new RunStartedEvent with displayName
    # Use model_copy if available, otherwise reconstruct
    if hasattr(event, "model_dump"):
        event_data = event.model_dump(by_alias=True, exclude_none=True)
        event_data["displayName"] = display_name
        return RunStartedEvent(**event_data)

    return event
