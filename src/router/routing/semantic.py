"""Semantic router using sentence-transformers for message routing."""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from router.config.agents import AgentConfig, AgentsConfig

logger = logging.getLogger(__name__)

# Security: Maximum message length to prevent DoS via large inputs
MAX_MESSAGE_LENGTH = 10000


@dataclass
class RouteMatch:
    """Result of a semantic routing match."""

    agent: AgentConfig
    score: float
    example: str  # The matched example utterance


class SemanticRouter:
    """Semantic router that matches messages to agents based on example utterances.

    Uses sentence-transformers to embed messages and compute cosine similarity
    against pre-embedded agent routing examples.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the semantic router.

        Args:
            model_name: Name of the sentence-transformers model to use.
        """
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._agents: list[AgentConfig] = []
        self._embeddings: NDArray[np.float32] | None = None
        self._example_to_agent: list[tuple[int, str]] = []  # (agent_index, example)

    @property
    def is_loaded(self) -> bool:
        """Check if the model and routes are loaded."""
        return self._model is not None and self._embeddings is not None

    def load_model(self) -> None:
        """Load the sentence-transformers model.

        This is separated from __init__ to allow for lazy loading and
        to make startup time measurement easier.
        """
        if self._model is not None:
            return

        logger.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        logger.info("Embedding model loaded successfully")

    def build_index(self, config: AgentsConfig) -> None:
        """Build the route index from agent configuration.

        Embeds all example utterances from all agents and stores them
        for fast similarity matching.

        Args:
            config: The agents configuration containing routing examples.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self._agents = list(config.agents)

        # Collect all examples with their agent mappings
        all_examples: list[str] = []
        self._example_to_agent = []

        for agent_idx, agent in enumerate(self._agents):
            # Skip agents without routing configuration or examples
            if agent.routing is None or not agent.routing.examples:
                continue
            for example in agent.routing.examples:
                all_examples.append(example)
                self._example_to_agent.append((agent_idx, example))

        if not all_examples:
            logger.warning("No routing examples found in configuration")
            self._embeddings = np.array([], dtype=np.float32)
            return

        # Embed all examples
        logger.info("Embedding %d routing examples", len(all_examples))
        embeddings = self._model.encode(
            all_examples,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._embeddings = embeddings.astype(np.float32)
        logger.info("Route index built with %d embeddings", len(self._embeddings))

    def match(self, message: str) -> list[RouteMatch]:
        """Find matching agents for a message.

        Returns all agents whose similarity score exceeds their threshold,
        sorted by score (highest first), then by priority (lowest first).

        Args:
            message: The user message to route.

        Returns:
            List of RouteMatch objects, sorted by score and priority.
            Empty list if no matches above threshold.

        Raises:
            RuntimeError: If router is not initialized.
            ValueError: If message is empty, None, or exceeds maximum length.
        """
        if not self.is_loaded:
            raise RuntimeError("Router not initialized. Call load_model() and build_index().")

        if not message or not message.strip():
            raise ValueError("Message cannot be empty")

        if len(message) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters")

        if self._embeddings is None or len(self._embeddings) == 0:
            logger.debug("No embeddings in index, returning empty matches")
            return []

        logger.debug("Matching message: %s", message[:100])

        # Embed the query
        query_embedding = self._model.encode(
            message,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Compute cosine similarities (embeddings are normalized, so dot product = cosine)
        similarities = np.dot(self._embeddings, query_embedding)

        # Find best match per agent (agent can have multiple examples)
        agent_best_scores: dict[int, tuple[float, str]] = {}

        for idx, similarity in enumerate(similarities):
            agent_idx, example = self._example_to_agent[idx]
            current_best = agent_best_scores.get(agent_idx)
            if current_best is None or similarity > current_best[0]:
                agent_best_scores[agent_idx] = (float(similarity), example)

        # Filter by threshold and create matches
        matches: list[RouteMatch] = []
        for agent_idx, (score, example) in agent_best_scores.items():
            agent = self._agents[agent_idx]
            # Skip agents without routing (they shouldn't be in the index anyway)
            if agent.routing is None:
                continue
            if score >= agent.routing.threshold:
                matches.append(RouteMatch(agent=agent, score=score, example=example))

        # Sort by score (descending), then by priority (ascending)
        # Agents without routing default to priority 1 (though they shouldn't be in matches)
        matches.sort(
            key=lambda m: (-m.score, m.agent.routing.priority if m.agent.routing else 1)
        )

        if matches:
            logger.debug(
                "Found %d matches, best: %s (score=%.3f)",
                len(matches),
                matches[0].agent.id,
                matches[0].score,
            )
        else:
            logger.debug("No matches above threshold")

        return matches

    def match_best(self, message: str) -> RouteMatch | None:
        """Find the single best matching agent for a message.

        This is a convenience method that returns the top match.

        Args:
            message: The user message to route.

        Returns:
            The best RouteMatch, or None if no matches above threshold.
        """
        matches = self.match(message)
        return matches[0] if matches else None

    def compute_similarity(self, message: str, agent: AgentConfig) -> float:
        """Compute similarity between a message and an agent's examples.

        This is useful for topic drift detection. Uses cached embeddings
        when the agent is in the index for better performance.

        Args:
            message: The user message.
            agent: The agent to compare against.

        Returns:
            Maximum similarity score across all agent examples.

        Raises:
            RuntimeError: If router is not initialized.
            ValueError: If message is empty, None, or exceeds maximum length.
        """
        if not self.is_loaded:
            raise RuntimeError("Router not initialized. Call load_model() and build_index().")

        if not message or not message.strip():
            raise ValueError("Message cannot be empty")

        if len(message) > MAX_MESSAGE_LENGTH:
            raise ValueError(f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters")

        if agent.routing is None or not agent.routing.examples:
            return 0.0

        # Embed the query
        query_embedding = self._model.encode(
            message,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Try to use cached embeddings if agent is in the index
        agent_idx = None
        for idx, a in enumerate(self._agents):
            if a.id == agent.id:
                agent_idx = idx
                break

        if agent_idx is not None and self._embeddings is not None:
            # Use cached embeddings - find indices for this agent's examples
            example_indices = [
                i for i, (idx, _) in enumerate(self._example_to_agent) if idx == agent_idx
            ]
            if example_indices:
                example_embeddings = self._embeddings[example_indices]
                similarities = np.dot(example_embeddings, query_embedding)
                return float(np.max(similarities))

        # Fallback: embed agent examples (for agents not in index)
        example_embeddings = self._model.encode(
            agent.routing.examples,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Compute similarities and return max
        similarities = np.dot(example_embeddings, query_embedding)
        return float(np.max(similarities))
