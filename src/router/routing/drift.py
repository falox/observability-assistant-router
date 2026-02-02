"""Topic drift detection for sticky sessions."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from router.config import AgentConfig
    from router.routing.semantic import SemanticRouter

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of topic drift detection."""

    drifted: bool
    similarity_score: float
    threshold: float


def detect_topic_drift(
    message: str,
    agent: "AgentConfig",
    semantic_router: "SemanticRouter",
    drift_threshold: float,
) -> DriftResult:
    """Detect if a message has drifted from the current agent's topic.

    Topic drift occurs when a new message is no longer semantically
    similar to the agent's routing examples. This triggers re-routing
    to find a better-matched agent.

    Args:
        message: The user message.
        agent: The agent currently handling the conversation.
        semantic_router: The semantic router for computing similarity.
        drift_threshold: Similarity threshold below which drift is detected.

    Returns:
        DriftResult with drifted flag and scores.
    """
    try:
        similarity = semantic_router.compute_similarity(message, agent)

        drifted = similarity < drift_threshold

        if drifted:
            logger.info(
                "Topic drift detected (agent=%s, score=%.3f, threshold=%.3f)",
                agent.id,
                similarity,
                drift_threshold,
            )
        else:
            logger.debug(
                "No topic drift (agent=%s, score=%.3f, threshold=%.3f)",
                agent.id,
                similarity,
                drift_threshold,
            )

        return DriftResult(
            drifted=drifted,
            similarity_score=similarity,
            threshold=drift_threshold,
        )

    except (ValueError, RuntimeError) as e:
        # On error, assume drift to allow re-routing
        logger.warning("Error computing drift, assuming drifted: %s", e)
        return DriftResult(
            drifted=True,
            similarity_score=0.0,
            threshold=drift_threshold,
        )
