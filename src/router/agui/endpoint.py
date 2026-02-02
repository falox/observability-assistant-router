"""AG-UI protocol endpoint for the router.

This module provides the FastAPI router that handles incoming AG-UI protocol
requests, routes them to the appropriate agent via semantic matching, and
streams responses back to clients using Server-Sent Events (SSE).
"""

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from router.agents.proxy import AgentProxy, AgentProxyError
from router.agents.retry import RetryConfig
from router.agui.models import (
    BaseEvent,
    ChatRequest,
    EventType,
    RunErrorEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from router.config import AgentConfig, AgentsConfig
from router.config.settings import get_settings
from router.observability import AuditLogger, StreamBuffer
from router.observability.audit import configure_audit_logging
from router.routing import (
    LLMFallbackError,
    SemanticRouter,
    classify_with_llm,
    detect_topic_drift,
    parse_mention,
    strip_mention,
)
from router.session import SessionStore

logger = logging.getLogger(__name__)

# Create the FastAPI router for AG-UI endpoints
agui_router = APIRouter(prefix="/api/agui", tags=["AG-UI Protocol"])


def get_agents_config(request: Request) -> AgentsConfig:
    """Get agents configuration from app state.

    Args:
        request: The FastAPI request object.

    Returns:
        The agents configuration.

    Raises:
        HTTPException: If configuration is not loaded.
    """
    config = getattr(request.app.state, "agents_config", None)
    if config is None:
        raise HTTPException(status_code=503, detail="Configuration not loaded")
    return config


def get_semantic_router(request: Request) -> SemanticRouter:
    """Get semantic router from app state.

    Args:
        request: The FastAPI request object.

    Returns:
        The semantic router instance.

    Raises:
        HTTPException: If router is not initialized.
    """
    router = getattr(request.app.state, "semantic_router", None)
    if router is None:
        raise HTTPException(status_code=503, detail="Semantic router not initialized")
    return router


def get_agent_proxy(request: Request) -> AgentProxy:
    """Get or create agent proxy from app state.

    Args:
        request: The FastAPI request object.

    Returns:
        The agent proxy instance.
    """
    proxy = getattr(request.app.state, "agent_proxy", None)
    if proxy is None:
        settings = get_settings()
        retry_config = RetryConfig(
            max_attempts=settings.retry_attempts,
            base_delay_ms=settings.retry_backoff_ms,
        )
        proxy = AgentProxy(retry_config=retry_config)
        request.app.state.agent_proxy = proxy
    return proxy


def get_session_store(request: Request) -> SessionStore | None:
    """Get session store from app state.

    Args:
        request: The FastAPI request object.

    Returns:
        The session store or None if sessions are disabled.
    """
    return getattr(request.app.state, "session_store", None)


@agui_router.post("/chat")
async def chat(
    request: Request,
    chat_request: ChatRequest,
    authorization: Annotated[str | None, Header()] = None,
    x_request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None,
) -> EventSourceResponse:
    """AG-UI chat endpoint for routing messages to agents.

    This endpoint accepts AG-UI protocol requests, routes them to the appropriate
    agent based on semantic matching, and streams responses back as SSE events.

    Routing flow with sticky sessions:
    1. Check for @mention override -> route to mentioned agent
    2. Check existing session and topic drift -> stay with sticky agent or re-route
    3. Fall back to semantic routing for new conversations
    4. Create/update session with routed agent

    Args:
        request: The FastAPI request object.
        chat_request: The AG-UI chat request body.
        authorization: Optional Authorization header to forward.
        x_request_id: Optional request ID for tracing.

    Returns:
        EventSourceResponse: SSE stream of AG-UI events.
    """
    # Generate request ID if not provided
    request_id = x_request_id or str(uuid.uuid4())

    # Get settings for observability
    settings = get_settings()

    # Configure audit logging on first request
    configure_audit_logging(settings.audit_log_level)

    # Create audit logger for this request
    audit = AuditLogger(
        request_id=request_id,
        thread_id=chat_request.thread_id,
        enabled=settings.audit_enabled,
    )

    logger.info(
        "Received chat request (thread_id=%s, request_id=%s, messages=%d)",
        chat_request.thread_id,
        request_id,
        len(chat_request.messages),
    )

    # Extract user message preview for audit log
    user_message_preview = _extract_user_message(chat_request)

    # Log request received
    audit.log_request_received(
        message_count=len(chat_request.messages),
        has_authorization=authorization is not None,
        user_message_preview=user_message_preview,
    )

    # Get dependencies from app state
    agents_config = get_agents_config(request)
    semantic_router = get_semantic_router(request)
    agent_proxy = get_agent_proxy(request)
    session_store = get_session_store(request)

    # Build headers to forward
    headers: dict[str, str] = {"X-Request-ID": request_id}
    if authorization:
        headers["Authorization"] = authorization

    # Extract the last user message for routing
    user_message = _extract_user_message(chat_request)
    if not user_message:
        logger.warning("No user message found in request")
        raise HTTPException(status_code=400, detail="No user message found")

    # Route the message using sticky session logic
    agent, routing_info = await _route_with_sessions(
        user_message=user_message,
        thread_id=chat_request.thread_id,
        agents_config=agents_config,
        semantic_router=semantic_router,
        session_store=session_store,
        headers=headers,
    )

    # Log routing decision
    audit.log_routing_decision(
        agent_id=agent.id,
        agent_name=agent.name,
        routing_method=routing_info.get("method", "unknown"),
        confidence_score=routing_info.get("score"),
        topic_drift_detected=routing_info.get("topic_drift", False),
    )

    # Strip @mentions from the request before forwarding to agent
    forwarding_request = _strip_mentions_from_request(chat_request)

    # Get default agent for fallback
    default_agent = agents_config.get_default_agent()

    # Create a run_id for this request
    run_id = str(uuid.uuid4())

    # Create stream buffer for message reassembly
    stream_buffer = (
        StreamBuffer(
            thread_id=chat_request.thread_id,
            run_id=run_id,
            request_id=request_id,
            max_content_size=settings.stream_buffer_max_size,
        )
        if settings.stream_buffer_enabled
        else None
    )

    # Create the SSE event generator with fallback support
    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        # Log agent forwarding
        audit.log_agent_forwarded(
            agent_id=agent.id,
            agent_protocol=agent.protocol.value,
            attempt_number=1,
        )

        # Log stream started
        audit.log_stream_started(run_id=run_id)

        # Try primary agent first
        primary_failed = False
        failure_context: str | None = None

        try:
            async for event in agent_proxy.forward_request(agent, forwarding_request, headers):
                # Encode event to SSE format (EventSourceResponse adds "data: " prefix)
                event_type = _get_event_type_str(event)
                yield {
                    "event": event_type,
                    "data": event.model_dump_json(by_alias=True, exclude_none=True),
                }
            # Success - return
            return

        except AgentProxyError as e:
            # Primary agent failed - prepare for fallback
            primary_failed = True
            failure_context = (
                f"Agent '{e.agent_name or agent.name}' unavailable after {e.attempts} attempt(s). "
            )
            logger.warning("Primary agent failed, attempting fallback to default agent: %s", e)

            # Log agent error
            audit.log_agent_error(
                agent_id=agent.id,
                error_message=str(e),
                is_retryable=e.is_retryable,
                attempt_number=e.attempts,
            )

        except Exception as e:
            # Unexpected error - prepare for fallback
            primary_failed = True
            failure_context = f"Agent '{agent.name}' encountered an error. "
            logger.exception("Unexpected error with primary agent, attempting fallback: %s", e)

            # Log agent error
            audit.log_agent_error(
                agent_id=agent.id,
                error_message=str(e),
            )

        # Fallback to default agent if primary failed (and primary wasn't already the default)
        if primary_failed and not agents_config.is_default_agent(agent):
            # Log fallback triggered
            audit.log_fallback_triggered(
                original_agent_id=agent.id,
                fallback_agent_id=default_agent.id,
                reason=failure_context or "Unknown error",
            )

            try:
                logger.info("Falling back to default agent (original=%s)", agent.name)

                # Yield fallback context as a text message before the response
                async for event in _yield_fallback_context(failure_context):
                    event_type = _get_event_type_str(event)
                    yield {
                        "event": event_type,
                        "data": event.model_dump_json(by_alias=True, exclude_none=True),
                    }

                # Log agent forwarding for fallback
                audit.log_agent_forwarded(
                    agent_id=default_agent.id,
                    agent_protocol=default_agent.protocol.value,
                    attempt_number=1,
                )

                # Forward to default agent
                async for event in agent_proxy.forward_request(
                    default_agent, forwarding_request, headers
                ):
                    event_type = _get_event_type_str(event)
                    yield {
                        "event": event_type,
                        "data": event.model_dump_json(by_alias=True, exclude_none=True),
                    }
                return

            except AgentProxyError as fallback_error:
                logger.error("Fallback to default agent also failed: %s", fallback_error)

                audit.log_agent_error(
                    agent_id=default_agent.id,
                    error_message=str(fallback_error),
                    is_retryable=fallback_error.is_retryable,
                    attempt_number=fallback_error.attempts,
                )

                fallback_msg = f"Fallback: {fallback_error}"
                error_event = RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message=f"All agents unavailable. Primary: {failure_context}{fallback_msg}",
                )
                yield {
                    "event": EventType.RUN_ERROR.value,
                    "data": error_event.model_dump_json(by_alias=True, exclude_none=True),
                }

            except Exception as fallback_error:
                logger.exception("Unexpected error during fallback: %s", fallback_error)

                audit.log_agent_error(
                    agent_id=default_agent.id,
                    error_message=str(fallback_error),
                )

                error_event = RunErrorEvent(
                    type=EventType.RUN_ERROR,
                    message="Internal server error during fallback",
                )
                yield {
                    "event": EventType.RUN_ERROR.value,
                    "data": error_event.model_dump_json(by_alias=True, exclude_none=True),
                }
        elif primary_failed:
            # Primary was already the default agent, no fallback available
            error_event = RunErrorEvent(
                type=EventType.RUN_ERROR,
                message=f"Default agent unavailable: {failure_context}",
            )
            yield {
                "event": EventType.RUN_ERROR.value,
                "data": error_event.model_dump_json(by_alias=True, exclude_none=True),
            }

    # Wrap event generator with stream buffer if enabled
    async def buffered_event_generator() -> AsyncGenerator[dict[str, str], None]:
        if stream_buffer:
            async for event in stream_buffer.process_stream(event_generator()):
                yield event

            # Log message completion after stream ends
            buffered_message = stream_buffer.get_message()
            if buffered_message and buffered_message.complete:
                audit.log_message_complete(buffered_message)
        else:
            async for event in event_generator():
                yield event

    return EventSourceResponse(
        buffered_event_generator(),
        media_type="text/event-stream",
        headers={"X-Request-ID": request_id},
        sep="\n",
    )


async def _route_with_sessions(
    user_message: str,
    thread_id: str,
    agents_config: AgentsConfig,
    semantic_router: SemanticRouter,
    session_store: SessionStore | None,
    headers: dict[str, str] | None = None,
) -> tuple[AgentConfig, dict[str, any]]:
    """Route a message with sticky session support.

    Routing flow:
    1. Check for @mention override -> route to mentioned agent
    2. Check existing session and topic drift -> stay with sticky agent or re-route
    3. Semantic routing for new conversations
    4. LLM fallback classification if no semantic match
    5. Default agent as ultimate fallback
    6. Create/update session with routed agent

    Args:
        user_message: The user's message content.
        thread_id: The conversation thread ID.
        agents_config: The agents configuration.
        semantic_router: The semantic router for matching.
        session_store: The session store (None if sessions disabled).
        headers: Optional headers to forward to LLM fallback.

    Returns:
        Tuple of (AgentConfig, routing_info dict) where routing_info contains:
        - method: The routing method used
        - score: The confidence score (if applicable)
        - topic_drift: Whether topic drift was detected
    """
    agent: AgentConfig | None = None
    routing_method = "semantic"
    routing_score: float | None = None
    topic_drift_detected = False

    # Step 1: Check for @mention override
    mentioned_handle = parse_mention(user_message)
    if mentioned_handle:
        # Try to find agent by handle (includes default agent since it's in the agents array)
        agent = agents_config.get_agent_by_handle(mentioned_handle)
        if agent:
            routing_method = "mention"
            logger.info(
                "Routing via @mention (handle=%s, agent=%s)",
                mentioned_handle,
                agent.name,
            )
        else:
            logger.warning(
                "Unknown @mention handle: %s, falling back to other routing",
                mentioned_handle,
            )

    # Step 2: Check existing session (if no @mention override)
    if agent is None and session_store is not None:
        session = session_store.get(thread_id)
        if session:
            # Found existing session - check for topic drift
            sticky_agent = agents_config.get_agent_by_id(session.agent_id)

            if sticky_agent:
                # Get drift threshold from config
                drift_threshold = agents_config.session.topic_drift_threshold

                # Check if topic has drifted
                drift_result = detect_topic_drift(
                    message=user_message,
                    agent=sticky_agent,
                    semantic_router=semantic_router,
                    drift_threshold=drift_threshold,
                )

                if not drift_result.drifted:
                    # Stay with sticky agent
                    agent = sticky_agent
                    routing_method = "sticky"
                    routing_score = drift_result.similarity_score
                    session_store.touch(thread_id)
                    logger.info(
                        "Using sticky session (thread_id=%s, agent=%s, score=%.3f)",
                        thread_id,
                        agent.name,
                        drift_result.similarity_score,
                    )
                else:
                    # Topic drifted - delete session and re-route
                    topic_drift_detected = True
                    session_store.delete(thread_id)
                    logger.info(
                        "Topic drift detected, re-routing (thread_id=%s, old_agent=%s)",
                        thread_id,
                        sticky_agent.name,
                    )

    # Step 3: Semantic routing (if no agent found yet)
    if agent is None:
        match = semantic_router.match_best(user_message)
        if match:
            agent = match.agent
            routing_method = "semantic"
            routing_score = match.score
            logger.info(
                "Routed via semantic matching (agent=%s, score=%.3f, example=%s)",
                agent.name,
                match.score,
                match.example[:50] if match.example else "N/A",
            )
        else:
            # No semantic match found - try LLM fallback classification
            default_agent = agents_config.get_default_agent()
            if agents_config.agents:
                try:
                    classified_agent = await classify_with_llm(
                        message=user_message,
                        agents=agents_config.agents,
                        default_agent_url=str(default_agent.url),
                        headers=headers,
                    )
                    if classified_agent:
                        agent = classified_agent
                        routing_method = "llm_fallback"
                        logger.info(
                            "Routed via LLM classification (agent=%s)",
                            agent.name,
                        )
                except LLMFallbackError as e:
                    logger.warning(
                        "LLM fallback classification failed, using default agent: %s",
                        e,
                    )

            # Ultimate fallback: use default agent
            if agent is None:
                agent = default_agent
                routing_method = "default"
                logger.info("No match found, using default agent")

    # Step 4: Update session (if sessions enabled)
    if session_store is not None and agents_config.session.sticky_enabled:
        session_store.set(
            thread_id=thread_id,
            agent_id=agent.id,
            agent_handle=agent.handles[0],  # Use primary handle for session
        )

    logger.debug(
        "Routing complete (thread_id=%s, agent=%s, method=%s)",
        thread_id,
        agent.id,
        routing_method,
    )

    routing_info = {
        "method": routing_method,
        "score": routing_score,
        "topic_drift": topic_drift_detected,
    }

    return agent, routing_info


def _extract_user_message(request: ChatRequest) -> str | None:
    """Extract the last user message content from a chat request.

    Args:
        request: The chat request.

    Returns:
        The user message content or None if not found.
    """
    for message in reversed(request.messages):
        if message.role == "user":
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Extract text from content items
                texts = []
                for item in content:
                    if hasattr(item, "text"):
                        texts.append(item.text)
                if texts:
                    return " ".join(texts)
    return None


def _get_event_type_str(event: BaseEvent) -> str:
    """Get the event type as a string for SSE formatting.

    Args:
        event: The AG-UI event.

    Returns:
        The event type as a string.
    """
    if hasattr(event.type, "value"):
        return event.type.value
    return str(event.type)


def _strip_mentions_from_request(request: ChatRequest) -> ChatRequest:
    """Strip @mentions from all user messages in a chat request.

    Creates a copy of the request with mentions removed from user messages,
    so the downstream agent receives clean message content.

    Args:
        request: The original chat request.

    Returns:
        A new ChatRequest with @mentions stripped from user messages.
    """
    stripped_messages = []
    for message in request.messages:
        if message.role == "user":
            # Create a copy of the message with stripped content
            content = message.content
            if isinstance(content, str):
                stripped_content = strip_mention(content)
            elif isinstance(content, list):
                # Strip mentions from text content items
                stripped_items = []
                for item in content:
                    if hasattr(item, "text"):
                        # Create a copy with stripped text
                        item_copy = item.model_copy(
                            update={"text": strip_mention(item.text)}
                        )
                        stripped_items.append(item_copy)
                    else:
                        stripped_items.append(item)
                stripped_content = stripped_items
            else:
                stripped_content = content

            # Create a copy of the message with the stripped content
            stripped_message = message.model_copy(update={"content": stripped_content})
            stripped_messages.append(stripped_message)
        else:
            stripped_messages.append(message)

    return request.model_copy(update={"messages": stripped_messages})


async def _yield_fallback_context(
    failure_context: str,
) -> AsyncGenerator[BaseEvent, None]:
    """Yield AG-UI events with fallback context information.

    This function generates a text message that informs the user about
    the fallback to the default agent, providing context about what failed.

    Args:
        failure_context: Description of what failed and triggered the fallback.

    Yields:
        BaseEvent: AG-UI text message events with the fallback notice.
    """
    fallback_message = f"[Notice: {failure_context}Routing to general assistant.]\n\n"
    message_id = str(uuid.uuid4())

    # Yield text message start
    yield TextMessageStartEvent(
        type=EventType.TEXT_MESSAGE_START,
        message_id=message_id,
        role="assistant",
    )

    # Yield the fallback context as content
    yield TextMessageContentEvent(
        type=EventType.TEXT_MESSAGE_CONTENT,
        message_id=message_id,
        delta=fallback_message,
    )

    # Yield text message end
    yield TextMessageEndEvent(
        type=EventType.TEXT_MESSAGE_END,
        message_id=message_id,
    )
