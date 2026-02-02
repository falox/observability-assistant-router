"""Routing module for semantic message routing."""

from router.routing.drift import DriftResult, detect_topic_drift
from router.routing.llm_fallback import (
    LLMFallbackError,
    build_classification_prompt,
    classify_with_llm,
    parse_llm_response,
)
from router.routing.mention import parse_mention, strip_mention, strip_mentions
from router.routing.semantic import RouteMatch, SemanticRouter

__all__ = [
    "SemanticRouter",
    "RouteMatch",
    "parse_mention",
    "strip_mention",
    "strip_mentions",
    "detect_topic_drift",
    "DriftResult",
    "classify_with_llm",
    "build_classification_prompt",
    "parse_llm_response",
    "LLMFallbackError",
]
