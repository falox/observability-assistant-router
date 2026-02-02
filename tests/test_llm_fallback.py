"""Test LLM fallback routing functionality."""

import pytest

from router.config import AgentConfig
from router.routing import (
    LLMFallbackError,
    build_classification_prompt,
    classify_with_llm,
    parse_llm_response,
)


class TestBuildClassificationPrompt:
    """Test classification prompt building."""

    @pytest.fixture
    def sample_agents(self):
        """Create sample agent configurations."""
        return [
            AgentConfig(
                id="troubleshooting-agent",
                name="Troubleshooting Agent",
                handles=["troubleshoot"],
                url="http://troubleshoot:8080",
                routing={"examples": ["test"]},
                description="Handles Kubernetes troubleshooting and debugging",
            ),
            AgentConfig(
                id="metrics-agent",
                name="Metrics Agent",
                handles=["metrics"],
                url="http://metrics:8080",
                routing={"examples": ["test"]},
                description="Handles Prometheus queries and metrics",
            ),
        ]

    def test_builds_prompt_with_message(self, sample_agents):
        """Test prompt includes the user message."""
        prompt = build_classification_prompt("Why is my pod crashing?", sample_agents)

        assert "Why is my pod crashing?" in prompt

    def test_builds_prompt_with_agent_ids(self, sample_agents):
        """Test prompt includes agent IDs."""
        prompt = build_classification_prompt("test message", sample_agents)

        assert "troubleshooting-agent" in prompt
        assert "metrics-agent" in prompt

    def test_builds_prompt_with_descriptions(self, sample_agents):
        """Test prompt includes agent descriptions."""
        prompt = build_classification_prompt("test message", sample_agents)

        assert "Handles Kubernetes troubleshooting" in prompt
        assert "Handles Prometheus queries" in prompt

    def test_truncates_long_messages(self, sample_agents):
        """Test long messages are truncated."""
        long_message = "x" * 1000
        prompt = build_classification_prompt(long_message, sample_agents)

        # Should be truncated with ellipsis
        assert "..." in prompt
        # Should not contain the full 1000 characters
        assert "x" * 1000 not in prompt

    def test_handles_empty_description(self):
        """Test handling of agents without descriptions."""
        agents = [
            AgentConfig(
                id="no-desc-agent",
                name="No Description Agent",
                handles=["nodesc"],
                url="http://nodesc:8080",
                routing={"examples": ["test"]},
                description="",
            ),
        ]

        prompt = build_classification_prompt("test message", agents)

        assert "no-desc-agent" in prompt
        assert "No description available" in prompt


class TestParseLLMResponse:
    """Test LLM response parsing."""

    @pytest.fixture
    def sample_agents(self):
        """Create sample agent configurations."""
        return [
            AgentConfig(
                id="troubleshooting-agent",
                name="Troubleshooting Agent",
                handles=["troubleshoot"],
                url="http://troubleshoot:8080",
                routing={"examples": ["test"]},
            ),
            AgentConfig(
                id="metrics-agent",
                name="Metrics Agent",
                handles=["metrics"],
                url="http://metrics:8080",
                routing={"examples": ["test"]},
            ),
        ]

    def test_parses_exact_agent_id(self, sample_agents):
        """Test parsing exact agent ID response."""
        agent = parse_llm_response("troubleshooting-agent", sample_agents)

        assert agent is not None
        assert agent.id == "troubleshooting-agent"

    def test_parses_agent_id_with_whitespace(self, sample_agents):
        """Test parsing agent ID with surrounding whitespace."""
        agent = parse_llm_response("  troubleshooting-agent  \n", sample_agents)

        assert agent is not None
        assert agent.id == "troubleshooting-agent"

    def test_parses_agent_id_case_insensitive(self, sample_agents):
        """Test case-insensitive matching."""
        agent = parse_llm_response("TROUBLESHOOTING-AGENT", sample_agents)

        assert agent is not None
        assert agent.id == "troubleshooting-agent"

    def test_parses_agent_id_with_quotes(self, sample_agents):
        """Test parsing agent ID wrapped in quotes."""
        agent = parse_llm_response('"troubleshooting-agent"', sample_agents)

        assert agent is not None
        assert agent.id == "troubleshooting-agent"

    def test_parses_agent_id_from_multiline(self, sample_agents):
        """Test parsing only first line of multiline response."""
        response = "troubleshooting-agent\nThis is the agent for troubleshooting"
        agent = parse_llm_response(response, sample_agents)

        assert agent is not None
        assert agent.id == "troubleshooting-agent"

    def test_finds_agent_id_in_text(self, sample_agents):
        """Test finding agent ID within longer text."""
        response = "I recommend using the troubleshooting-agent for this query."
        agent = parse_llm_response(response, sample_agents)

        assert agent is not None
        assert agent.id == "troubleshooting-agent"

    def test_returns_none_for_empty_response(self, sample_agents):
        """Test returns None for empty response."""
        assert parse_llm_response("", sample_agents) is None
        assert parse_llm_response("   ", sample_agents) is None

    def test_returns_none_for_unknown_agent(self, sample_agents):
        """Test returns None for unknown agent ID."""
        agent = parse_llm_response("unknown-agent", sample_agents)

        assert agent is None

    def test_returns_none_for_partial_match(self, sample_agents):
        """Test returns None for partial agent ID match."""
        # "troubleshooting" alone should not match "troubleshooting-agent"
        # because we use word boundary matching
        agent = parse_llm_response("troubleshooting", sample_agents)

        assert agent is None


class TestClassifyWithLLM:
    """Test LLM classification function."""

    @pytest.fixture
    def sample_agents(self):
        """Create sample agent configurations."""
        return [
            AgentConfig(
                id="troubleshooting-agent",
                name="Troubleshooting Agent",
                handles=["troubleshoot"],
                url="http://troubleshoot:8080",
                routing={"examples": ["test"]},
                description="Handles Kubernetes troubleshooting",
            ),
        ]

    @pytest.mark.asyncio
    async def test_classify_returns_none_for_empty_agents(self):
        """Test classification returns None when no agents configured."""
        agent = await classify_with_llm(
            message="test message",
            agents=[],
            default_agent_url="http://default:8080",
        )

        assert agent is None

    @pytest.mark.asyncio
    async def test_classify_raises_on_connection_error(self, sample_agents):
        """Test classification raises LLMFallbackError on connection error."""
        # Use a non-routable address to trigger connection error
        with pytest.raises(LLMFallbackError):
            await classify_with_llm(
                message="test message",
                agents=sample_agents,
                default_agent_url="http://192.0.2.1:9999",  # TEST-NET, non-routable
                timeout=0.5,  # Short timeout
            )


class TestExtractTextFromA2AResponse:
    """Test A2A response text extraction."""

    def test_extracts_from_artifacts(self):
        """Test extracting text from artifacts format."""
        from router.routing.llm_fallback import _extract_text_from_a2a_response

        response = {
            "result": {"artifacts": [{"parts": [{"kind": "text", "text": "test response"}]}]}
        }

        text = _extract_text_from_a2a_response(response)
        assert text == "test response"

    def test_extracts_from_message_format(self):
        """Test extracting text from message format."""
        from router.routing.llm_fallback import _extract_text_from_a2a_response

        response = {"result": {"message": {"parts": [{"kind": "text", "text": "message text"}]}}}

        text = _extract_text_from_a2a_response(response)
        assert text == "message text"

    def test_extracts_without_kind_field(self):
        """Test extracting text when kind field is missing."""
        from router.routing.llm_fallback import _extract_text_from_a2a_response

        response = {"result": {"artifacts": [{"parts": [{"text": "no kind field"}]}]}}

        text = _extract_text_from_a2a_response(response)
        assert text == "no kind field"

    def test_returns_none_for_empty_response(self):
        """Test returns None for empty response."""
        from router.routing.llm_fallback import _extract_text_from_a2a_response

        assert _extract_text_from_a2a_response({}) is None
        assert _extract_text_from_a2a_response({"result": {}}) is None

    def test_returns_none_for_invalid_format(self):
        """Test returns None for invalid response format."""
        from router.routing.llm_fallback import _extract_text_from_a2a_response

        response = {"result": {"unexpected": "format"}}
        assert _extract_text_from_a2a_response(response) is None
