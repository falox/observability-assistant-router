"""Tests for topic drift detection."""

import pytest

from router.config import AgentConfig, AgentProtocol, AgentRoutingConfig, AgentsConfig
from router.routing.drift import DriftResult, detect_topic_drift
from router.routing.semantic import SemanticRouter


@pytest.fixture
def agents_config():
    """Create a test agents config."""
    return AgentsConfig(
        default_agent={"id": "default"},
        agents=[
            AgentConfig(
                id="default",
                name="Default Agent",
                handles=["assistant"],
                url="http://default.local",
                protocol=AgentProtocol.A2A,
                routing=AgentRoutingConfig(
                    priority=100,
                    threshold=0.3,
                    examples=["general question", "help me"],
                ),
            ),
            AgentConfig(
                id="troubleshooting",
                name="Troubleshooting Agent",
                handles=["troubleshoot", "debug"],
                url="http://troubleshooting.local",
                protocol=AgentProtocol.A2A,
                routing=AgentRoutingConfig(
                    priority=1,
                    threshold=0.7,
                    examples=[
                        "Why is my pod crashing?",
                        "Debug the CrashLoopBackOff error",
                        "Help me fix this deployment failure",
                    ],
                ),
            ),
            AgentConfig(
                id="metrics",
                name="Metrics Agent",
                handles=["metrics", "prometheus"],
                url="http://metrics.local",
                protocol=AgentProtocol.A2A,
                routing=AgentRoutingConfig(
                    priority=2,
                    threshold=0.7,
                    examples=[
                        "Show me CPU usage",
                        "Query Prometheus metrics",
                        "Display memory consumption",
                    ],
                ),
            ),
        ],
    )


@pytest.fixture
def semantic_router(agents_config):
    """Create a test semantic router."""
    router = SemanticRouter()
    router.load_model()
    router.build_index(agents_config)
    return router


class TestDetectTopicDrift:
    """Tests for detect_topic_drift function."""

    def test_no_drift_on_topic_message(self, agents_config, semantic_router):
        """Test that on-topic messages don't trigger drift."""
        agent = agents_config.agents[1]  # troubleshooting

        result = detect_topic_drift(
            message="Why is my container restarting?",
            agent=agent,
            semantic_router=semantic_router,
            drift_threshold=0.4,  # Lower threshold to account for semantic similarity
        )

        assert isinstance(result, DriftResult)
        assert result.drifted is False
        assert result.similarity_score > 0.4
        assert result.threshold == 0.4

    def test_drift_on_off_topic_message(self, agents_config, semantic_router):
        """Test that off-topic messages trigger drift."""
        agent = agents_config.agents[1]  # troubleshooting

        result = detect_topic_drift(
            message="Show me the CPU metrics graph",
            agent=agent,
            semantic_router=semantic_router,
            drift_threshold=0.7,
        )

        assert isinstance(result, DriftResult)
        # This is a metrics query, should drift from troubleshooting
        assert result.drifted is True
        assert result.similarity_score < 0.7
        assert result.threshold == 0.7

    def test_drift_threshold_respected(self, agents_config, semantic_router):
        """Test that drift threshold is respected."""
        agent = agents_config.agents[1]  # troubleshooting

        # With high threshold, even somewhat related messages drift
        result_high = detect_topic_drift(
            message="Container health issues",
            agent=agent,
            semantic_router=semantic_router,
            drift_threshold=0.9,
        )

        # With low threshold, messages stay on topic
        result_low = detect_topic_drift(
            message="Container health issues",
            agent=agent,
            semantic_router=semantic_router,
            drift_threshold=0.3,
        )

        # Same score, different drift decisions
        assert result_high.drifted is True
        assert result_low.drifted is False
        # Scores should be the same
        assert abs(result_high.similarity_score - result_low.similarity_score) < 0.01

    def test_drift_on_completely_unrelated(self, agents_config, semantic_router):
        """Test drift on completely unrelated message."""
        agent = agents_config.agents[1]  # troubleshooting

        result = detect_topic_drift(
            message="What's the weather like today?",
            agent=agent,
            semantic_router=semantic_router,
            drift_threshold=0.5,
        )

        assert result.drifted is True
        assert result.similarity_score < 0.5

    def test_no_drift_on_exact_example(self, agents_config, semantic_router):
        """Test no drift when message matches an example exactly."""
        agent = agents_config.agents[1]  # troubleshooting

        result = detect_topic_drift(
            message="Why is my pod crashing?",
            agent=agent,
            semantic_router=semantic_router,
            drift_threshold=0.9,
        )

        assert result.drifted is False
        assert result.similarity_score > 0.95

    def test_drift_result_dataclass(self, agents_config, semantic_router):
        """Test DriftResult dataclass properties."""
        agent = agents_config.agents[1]  # troubleshooting

        result = detect_topic_drift(
            message="Test message",
            agent=agent,
            semantic_router=semantic_router,
            drift_threshold=0.5,
        )

        assert hasattr(result, "drifted")
        assert hasattr(result, "similarity_score")
        assert hasattr(result, "threshold")
        assert isinstance(result.drifted, bool)
        assert isinstance(result.similarity_score, float)
        assert isinstance(result.threshold, float)
