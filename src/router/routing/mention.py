"""@mention parser for agent override routing."""

import logging
import re

logger = logging.getLogger(__name__)

# Regex pattern for @mentions
# Matches @ followed by alphanumeric characters, hyphens, or underscores
# Case-insensitive matching is handled by lowercasing
MENTION_PATTERN = re.compile(r"@([a-zA-Z0-9_-]+)")


def parse_mention(message: str) -> str | None:
    """Parse the first @mention from a message.

    @mentions allow users to explicitly route to a specific agent,
    bypassing semantic routing. The first mention in the message wins.

    Examples:
        "@troubleshoot why is my pod crashing?" -> "troubleshoot"
        "@Metrics show CPU usage" -> "metrics" (case-insensitive)
        "@metrics @troubleshoot help" -> "metrics" (first wins)
        "Help me @troubleshoot this" -> "troubleshoot"
        "No mention here" -> None

    Args:
        message: The user message to parse.

    Returns:
        The handle (lowercase) if a mention is found, None otherwise.
    """
    if not message:
        return None

    match = MENTION_PATTERN.search(message)
    if match:
        handle = match.group(1).lower()
        logger.debug("Parsed @mention: %s", handle)
        return handle

    return None


def strip_mentions(message: str) -> str:
    """Remove all @mentions from a message.

    Useful for passing the message content to the agent without the mentions.
    The first @mention is used for routing, but the agent should receive
    clean message content without any @handle syntax.

    Examples:
        "@troubleshoot why is my pod crashing?" -> "why is my pod crashing?"
        "@metrics @prometheus show CPU usage" -> "show CPU usage"
        "Help me @troubleshoot this @debug issue" -> "Help me this issue"

    Args:
        message: The user message.

    Returns:
        The message with all @mentions removed and whitespace normalized.
    """
    if not message:
        return message

    # Remove all mentions and clean up whitespace
    result = MENTION_PATTERN.sub("", message)
    return " ".join(result.split())


# Keep old name as alias for backward compatibility
strip_mention = strip_mentions
