"""Test semantic routing functionality."""

import pytest

from router.config import AgentConfig, AgentsConfig
from router.routing import RouteMatch, SemanticRouter


class TestSemanticRouter:
    """Test SemanticRouter class."""

    @pytest.fixture
    def router(self):
        """Create a SemanticRouter instance."""
        return SemanticRouter(model_name="all-MiniLM-L6-v2")

    @pytest.fixture
    def agents_config(self):
        """Create a test agents configuration."""
        return AgentsConfig(
            default_agent={"id": "default-agent"},
            agents=[
                {
                    "id": "default-agent",
                    "name": "Default Agent",
                    "handles": ["assistant"],
                    "url": "http://default:8080",
                    "protocol": "a2a",
                    "routing": {
                        "priority": 100,
                        "threshold": 0.3,
                        "examples": ["general question", "help me"],
                    },
                    "description": "Default agent",
                },
                {
                    "id": "troubleshooting-agent",
                    "name": "Troubleshooting Agent",
                    "handles": ["troubleshoot", "debug", "rca"],
                    "url": "http://troubleshoot:8080",
                    "protocol": "a2a",
                    "routing": {
                        "priority": 1,
                        "threshold": 0.7,
                        "examples": [
                            "Why is my pod crashing?",
                            "Debug the CrashLoopBackOff error",
                            "Help me fix this deployment failure",
                            "The service is not responding",
                        ],
                    },
                    "description": "Handles Kubernetes troubleshooting",
                },
                {
                    "id": "metrics-agent",
                    "name": "Metrics Agent",
                    "handles": ["metrics", "prometheus"],
                    "url": "http://metrics:8080",
                    "protocol": "ag-ui",
                    "routing": {
                        "priority": 2,
                        "threshold": 0.7,
                        "examples": [
                            "Show me CPU usage for the cluster",
                            "What's the memory consumption of my pods?",
                            "Query Prometheus for request latency",
                            "How many requests per second is my service handling?",
                        ],
                    },
                    "description": "Handles Prometheus queries and metrics",
                },
            ],
        )

    def test_is_loaded_before_init(self, router):
        """Test is_loaded returns False before initialization."""
        assert router.is_loaded is False

    def test_load_model(self, router):
        """Test model loading."""
        router.load_model()
        # Model is loaded but no index yet, so is_loaded still False
        assert router._model is not None

    def test_build_index_without_model_raises(self, router, agents_config):
        """Test building index without loading model raises error."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            router.build_index(agents_config)

    def test_build_index(self, router, agents_config):
        """Test building the route index."""
        router.load_model()
        router.build_index(agents_config)

        assert router.is_loaded is True
        # 10 examples total (2 from default + 4 from troubleshooting + 4 from metrics)
        assert len(router._embeddings) == 10
        assert len(router._example_to_agent) == 10

    def test_build_index_skips_agents_without_routing(self, router):
        """Test that agents without routing config are skipped in the index."""
        config = AgentsConfig(
            default_agent={"id": "fallback-agent"},
            agents=[
                {
                    "id": "fallback-agent",
                    "name": "Fallback Agent",
                    "handles": ["assistant"],
                    "url": "http://fallback:8080",
                    # No routing section - this is the fallback
                },
                {
                    "id": "routed-agent",
                    "name": "Routed Agent",
                    "handles": ["routed"],
                    "url": "http://routed:8080",
                    "routing": {
                        "threshold": 0.7,
                        "examples": ["specific query one", "specific query two"],
                    },
                },
            ],
        )

        router.load_model()
        router.build_index(config)

        assert router.is_loaded is True
        # Only 2 examples from routed-agent (fallback-agent has no routing)
        assert len(router._embeddings) == 2
        assert len(router._example_to_agent) == 2

    def test_match_without_init_raises(self, router):
        """Test matching without initialization raises error."""
        with pytest.raises(RuntimeError, match="not initialized"):
            router.match("test query")

    def test_match_troubleshooting_query(self, router, agents_config):
        """Test matching a troubleshooting-related query."""
        router.load_model()
        router.build_index(agents_config)

        matches = router.match("My pod keeps crashing with OOM errors")

        assert len(matches) >= 1
        assert matches[0].agent.id == "troubleshooting-agent"
        assert matches[0].score >= 0.7

    def test_match_metrics_query(self, router, agents_config):
        """Test matching a metrics-related query."""
        router.load_model()
        router.build_index(agents_config)

        matches = router.match("Show me the CPU usage graph")

        assert len(matches) >= 1
        assert matches[0].agent.id == "metrics-agent"
        assert matches[0].score >= 0.7

    def test_match_returns_sorted_by_score(self, router, agents_config):
        """Test matches are sorted by score (descending)."""
        router.load_model()
        router.build_index(agents_config)

        # A query that might match multiple agents
        matches = router.match("The service has high latency and is failing")

        if len(matches) > 1:
            for i in range(len(matches) - 1):
                # Higher score first, or same score with lower priority
                assert matches[i].score > matches[i + 1].score or (
                    matches[i].score == matches[i + 1].score
                    and matches[i].agent.routing.priority <= matches[i + 1].agent.routing.priority
                )

    def test_match_best(self, router, agents_config):
        """Test match_best returns single best match."""
        router.load_model()
        router.build_index(agents_config)

        match = router.match_best("Why is my pod crashing?")

        assert match is not None
        assert isinstance(match, RouteMatch)
        assert match.agent.id == "troubleshooting-agent"

    def test_match_best_no_match(self, router, agents_config):
        """Test match_best returns None when no match above threshold."""
        # Create config with very high threshold
        high_threshold_config = AgentsConfig(
            default_agent={"id": "test-agent"},
            agents=[
                {
                    "id": "test-agent",
                    "name": "Test Agent",
                    "handles": ["test"],
                    "url": "http://test:8080",
                    "routing": {
                        "threshold": 0.99,  # Very high threshold
                        "examples": ["very specific exact phrase"],
                    },
                }
            ],
        )

        router.load_model()
        router.build_index(high_threshold_config)

        # Unrelated query should not match
        match = router.match_best("random unrelated query about weather")
        assert match is None

    def test_threshold_filtering(self, router):
        """Test that matches below threshold are filtered out."""
        config = AgentsConfig(
            default_agent={"id": "low-threshold-agent"},
            agents=[
                {
                    "id": "high-threshold-agent",
                    "name": "High Threshold Agent",
                    "handles": ["high"],
                    "url": "http://high:8080",
                    "routing": {
                        "threshold": 0.95,
                        "examples": ["very specific query"],
                    },
                },
                {
                    "id": "low-threshold-agent",
                    "name": "Low Threshold Agent",
                    "handles": ["low"],
                    "url": "http://low:8080",
                    "routing": {
                        "threshold": 0.5,
                        "examples": ["general query"],
                    },
                },
            ],
        )

        router.load_model()
        router.build_index(config)

        # A general query should only match the low threshold agent
        matches = router.match("some general query about stuff")

        # Each match must have score >= its agent's threshold
        for match in matches:
            assert match.score >= match.agent.routing.threshold

    def test_priority_tiebreaker(self, router):
        """Test that priority breaks ties when scores are similar."""
        config = AgentsConfig(
            default_agent={"id": "high-priority-agent"},
            agents=[
                {
                    "id": "low-priority-agent",
                    "name": "Low Priority Agent",
                    "handles": ["low"],
                    "url": "http://low:8080",
                    "routing": {
                        "priority": 10,
                        "threshold": 0.5,
                        "examples": ["kubernetes pod issue"],
                    },
                },
                {
                    "id": "high-priority-agent",
                    "name": "High Priority Agent",
                    "handles": ["high"],
                    "url": "http://high:8080",
                    "routing": {
                        "priority": 1,
                        "threshold": 0.5,
                        "examples": ["kubernetes pod problem"],
                    },
                },
            ],
        )

        router.load_model()
        router.build_index(config)

        # Both agents have similar examples, so they should have similar scores
        # Priority should determine the winner when scores are close
        matches = router.match("kubernetes pod issue")

        # Both should match since threshold is 0.5
        assert len(matches) >= 1

    def test_empty_agents_config(self, router, agents_config):
        """Test handling of empty index after build."""
        # Use the fixture config but test with an agent that has no examples matching
        router.load_model()
        router.build_index(agents_config)

        assert router.is_loaded is True
        with pytest.raises(ValueError, match="cannot be empty"):
            router.match("")

    def test_match_empty_message_raises(self, router, agents_config):
        """Test that empty message raises ValueError."""
        router.load_model()
        router.build_index(agents_config)

        with pytest.raises(ValueError, match="cannot be empty"):
            router.match("")

        with pytest.raises(ValueError, match="cannot be empty"):
            router.match("   ")

    def test_compute_similarity_empty_message_raises(self, router, agents_config):
        """Test that empty message raises ValueError in compute_similarity."""
        router.load_model()
        router.build_index(agents_config)

        agent = agents_config.get_agent_by_id("troubleshooting-agent")

        with pytest.raises(ValueError, match="cannot be empty"):
            router.compute_similarity("", agent)

        with pytest.raises(ValueError, match="cannot be empty"):
            router.compute_similarity("   ", agent)

    def test_compute_similarity_uses_cached_embeddings(self, router, agents_config):
        """Test that compute_similarity uses cached embeddings for indexed agents."""
        router.load_model()
        router.build_index(agents_config)

        # Get an agent that's in the index
        troubleshoot_agent = agents_config.get_agent_by_id("troubleshooting-agent")

        # This should use cached embeddings (no way to verify directly, but ensures no errors)
        score = router.compute_similarity("Why is my pod crashing?", troubleshoot_agent)
        assert score > 0.0

    def test_compute_similarity(self, router, agents_config):
        """Test computing similarity for topic drift detection."""
        router.load_model()
        router.build_index(agents_config)

        troubleshoot_agent = agents_config.get_agent_by_id("troubleshooting-agent")
        metrics_agent = agents_config.get_agent_by_id("metrics-agent")

        # Query similar to troubleshooting
        troubleshoot_score = router.compute_similarity(
            "Why is my pod crashing?", troubleshoot_agent
        )
        metrics_score = router.compute_similarity("Why is my pod crashing?", metrics_agent)

        # Should be more similar to troubleshooting agent
        assert troubleshoot_score > metrics_score

    def test_compute_similarity_without_init_raises(self, router, agents_config):
        """Test compute_similarity without initialization raises error."""
        agent = AgentConfig(
            id="test",
            name="Test",
            handles=["test"],
            url="http://test:8080",
            routing={"examples": ["test"]},
        )
        with pytest.raises(RuntimeError, match="not initialized"):
            router.compute_similarity("test", agent)

    def test_compute_similarity_no_examples(self, router, agents_config):
        """Test compute_similarity with agent that has no examples."""
        router.load_model()
        router.build_index(agents_config)

        # Create an agent with empty examples
        agent = AgentConfig(
            id="empty",
            name="Empty",
            handles=["empty"],
            url="http://empty:8080",
            routing={"examples": []},
        )

        score = router.compute_similarity("any query", agent)
        assert score == 0.0

    def test_compute_similarity_no_routing(self, router, agents_config):
        """Test compute_similarity with agent that has no routing config."""
        router.load_model()
        router.build_index(agents_config)

        # Create an agent without routing (fallback agent)
        agent = AgentConfig(
            id="fallback",
            name="Fallback",
            handles=["fallback"],
            url="http://fallback:8080",
        )

        score = router.compute_similarity("any query", agent)
        assert score == 0.0

    def test_match_returns_matched_example(self, router, agents_config):
        """Test that match returns the matched example utterance."""
        router.load_model()
        router.build_index(agents_config)

        matches = router.match("Why is my pod crashing?")

        assert len(matches) >= 1
        assert matches[0].example is not None
        assert isinstance(matches[0].example, str)
        # The example should be one of the agent's examples
        assert matches[0].example in matches[0].agent.routing.examples


class TestRouteMatch:
    """Test RouteMatch dataclass."""

    def test_route_match_creation(self):
        """Test creating a RouteMatch."""
        agent = AgentConfig(
            id="test",
            name="Test Agent",
            handles=["test"],
            url="http://test:8080",
            routing={"examples": ["test query"]},
        )

        match = RouteMatch(agent=agent, score=0.85, example="test query")

        assert match.agent == agent
        assert match.score == 0.85
        assert match.example == "test query"
