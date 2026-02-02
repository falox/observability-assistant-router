"""LLM fallback for routing ambiguous messages.

When semantic routing doesn't find a match above the threshold,
this module uses the default agent's LLM to classify the message
and determine which specialist agent should handle it.
"""

import logging
import re
import uuid

import httpx

from router.config import AgentConfig

logger = logging.getLogger(__name__)

# Maximum characters for message in classification prompt
MAX_MESSAGE_LENGTH = 500

# System prompt for classification (not currently used, but available for future)
CLASSIFICATION_SYSTEM_PROMPT = (
    "You are a message router that classifies user queries "
    "to the most appropriate specialist agent. "
    "Based on the user's query and the available agents with their descriptions, "
    "determine which agent should handle the request. "
    "IMPORTANT: Respond with ONLY the agent ID (e.g., 'troubleshooting-agent') "
    "on a single line. Do not include any explanation or additional text."
)

# Template for the classification prompt
CLASSIFICATION_PROMPT_TEMPLATE = """User query: {message}

Available specialist agents:
{agent_list}

Which agent should handle this query? Respond with ONLY the agent ID."""


class LLMFallbackError(Exception):
    """Raised when LLM fallback classification fails."""

    pass


def build_classification_prompt(
    message: str,
    agents: list[AgentConfig],
) -> str:
    """Build a classification prompt for the LLM.

    Args:
        message: The user's message to classify.
        agents: List of available agents with descriptions.

    Returns:
        The formatted classification prompt.
    """
    # Truncate message if too long
    truncated_message = message[:MAX_MESSAGE_LENGTH]
    if len(message) > MAX_MESSAGE_LENGTH:
        truncated_message += "..."

    # Build agent list with descriptions
    agent_lines = []
    for agent in agents:
        description = agent.description.strip() if agent.description else "No description available"
        agent_lines.append(f"- {agent.id}: {description}")

    agent_list = "\n".join(agent_lines)

    return CLASSIFICATION_PROMPT_TEMPLATE.format(
        message=truncated_message,
        agent_list=agent_list,
    )


def parse_llm_response(
    response: str,
    agents: list[AgentConfig],
) -> AgentConfig | None:
    """Parse the LLM response to extract the agent ID.

    The LLM is expected to respond with just the agent ID.
    This function validates that the ID matches a known agent.

    Args:
        response: The LLM's response text.
        agents: List of available agents.

    Returns:
        The matched AgentConfig or None if no match found.
    """
    if not response or not response.strip():
        logger.warning("Empty LLM response for classification")
        return None

    # Clean the response - take first line, strip whitespace and quotes
    response_text = response.strip().split("\n")[0].strip()
    response_text = response_text.strip("\"'")

    # Try exact match first
    for agent in agents:
        if agent.id == response_text:
            logger.debug("LLM classification matched agent: %s", agent.id)
            return agent

    # Try case-insensitive match
    response_lower = response_text.lower()
    for agent in agents:
        if agent.id.lower() == response_lower:
            logger.debug("LLM classification matched agent (case-insensitive): %s", agent.id)
            return agent

    # Try to find agent ID within the response (in case LLM added extra text)
    for agent in agents:
        # Use word boundary to avoid partial matches
        pattern = rf"\b{re.escape(agent.id)}\b"
        if re.search(pattern, response_text, re.IGNORECASE):
            logger.debug("LLM classification found agent ID in response: %s", agent.id)
            return agent

    logger.warning(
        "LLM response '%s' did not match any known agent",
        response_text[:100],
    )
    return None


async def classify_with_llm(
    message: str,
    agents: list[AgentConfig],
    default_agent_url: str,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> AgentConfig | None:
    """Use the default agent's LLM to classify a message.

    This function sends a classification prompt to the default agent
    and parses the response to determine which specialist agent
    should handle the original message.

    Args:
        message: The user's message to classify.
        agents: List of available specialist agents.
        default_agent_url: URL of the default agent.
        headers: Optional headers to forward (Authorization, etc.).
        timeout: Request timeout in seconds.

    Returns:
        The classified AgentConfig or None if classification fails.

    Raises:
        LLMFallbackError: If the LLM request fails.
    """
    if not agents:
        logger.debug("No agents configured for LLM classification")
        return None

    # Build the classification prompt
    prompt = build_classification_prompt(message, agents)

    logger.debug(
        "Calling LLM fallback for classification (message_len=%d, agents=%d)",
        len(message),
        len(agents),
    )

    # Create request payload for A2A format
    # The default agent uses A2A protocol
    context_id = str(uuid.uuid4())
    request_payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [{"kind": "text", "text": prompt}],
            },
            "configuration": {
                "acceptedOutputModes": ["text"],
            },
        },
    }

    # Add context ID if available
    request_payload["params"]["contextId"] = context_id

    # Prepare headers
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        # Forward authorization but not request ID (this is a separate request)
        if "Authorization" in headers:
            request_headers["Authorization"] = headers["Authorization"]

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                default_agent_url,
                json=request_payload,
                headers=request_headers,
            )
            response.raise_for_status()

            result = response.json()

            # Extract text from A2A response
            llm_text = _extract_text_from_a2a_response(result)

            if not llm_text:
                logger.warning("No text content in LLM classification response")
                return None

            # Parse the response to find the agent
            return parse_llm_response(llm_text, agents)

    except httpx.HTTPStatusError as e:
        logger.error(
            "LLM classification request failed with status %d: %s",
            e.response.status_code,
            e.response.text[:200] if e.response.text else "No response body",
        )
        raise LLMFallbackError(f"LLM classification failed: HTTP {e.response.status_code}") from e

    except httpx.RequestError as e:
        logger.error("LLM classification request error: %s", e)
        raise LLMFallbackError(f"LLM classification request failed: {e}") from e

    except Exception as e:
        logger.exception("Unexpected error during LLM classification: %s", e)
        raise LLMFallbackError(f"LLM classification failed: {e}") from e


def _extract_text_from_a2a_response(response: dict) -> str | None:
    """Extract text content from an A2A JSON-RPC response.

    Args:
        response: The A2A JSON-RPC response.

    Returns:
        The extracted text or None if not found.
    """
    try:
        # A2A response format: {"result": {"artifacts": [{"parts": [{"text": "..."}]}]}}
        result = response.get("result", {})

        # Check for artifacts (task completion)
        artifacts = result.get("artifacts", [])
        for artifact in artifacts:
            parts = artifact.get("parts", [])
            for part in parts:
                if part.get("kind") == "text" and "text" in part:
                    return part["text"]
                # Also try without kind check
                if "text" in part:
                    return part["text"]

        # Check for message format
        message = result.get("message", {})
        parts = message.get("parts", [])
        for part in parts:
            if part.get("kind") == "text" and "text" in part:
                return part["text"]
            if "text" in part:
                return part["text"]

        # Check for direct text field
        if "text" in result:
            return result["text"]

        logger.debug("Could not extract text from A2A response: %s", response)
        return None

    except (KeyError, TypeError, AttributeError) as e:
        logger.warning("Error parsing A2A response: %s", e)
        return None
