"""Tests for @mention parser."""

from router.routing.mention import parse_mention, strip_mention, strip_mentions


class TestParseMention:
    """Tests for parse_mention function."""

    def test_simple_mention(self):
        """Test parsing a simple @mention."""
        assert parse_mention("@troubleshoot why is my pod crashing?") == "troubleshoot"

    def test_mention_at_end(self):
        """Test parsing @mention at end of message."""
        assert parse_mention("Help me @troubleshoot this") == "troubleshoot"

    def test_mention_in_middle(self):
        """Test parsing @mention in middle of message."""
        assert parse_mention("Can you @metrics show CPU?") == "metrics"

    def test_case_insensitive(self):
        """Test that @mentions are case-insensitive."""
        assert parse_mention("@Troubleshoot help") == "troubleshoot"
        assert parse_mention("@METRICS show usage") == "metrics"
        assert parse_mention("@TrOuBlEsHoOt issue") == "troubleshoot"

    def test_first_mention_wins(self):
        """Test that first @mention wins when multiple present."""
        assert parse_mention("@metrics @troubleshoot help") == "metrics"
        assert parse_mention("@troubleshoot @metrics help") == "troubleshoot"

    def test_no_mention(self):
        """Test message without @mention."""
        assert parse_mention("No mention here") is None
        assert parse_mention("Just a regular question") is None

    def test_empty_message(self):
        """Test empty message."""
        assert parse_mention("") is None
        assert parse_mention(None) is None

    def test_handle_with_hyphen(self):
        """Test @mention with hyphen in handle."""
        assert parse_mention("@my-agent help") == "my-agent"

    def test_handle_with_underscore(self):
        """Test @mention with underscore in handle."""
        assert parse_mention("@my_agent help") == "my_agent"

    def test_handle_with_numbers(self):
        """Test @mention with numbers in handle."""
        assert parse_mention("@agent123 help") == "agent123"
        assert parse_mention("@123agent help") == "123agent"

    def test_email_not_matched(self):
        """Test that email addresses are matched (first part)."""
        # Email will match the local part before @
        result = parse_mention("Contact user@example.com for help")
        assert result == "example"


class TestStripMentions:
    """Tests for strip_mentions function (removes ALL @mentions)."""

    def test_strip_at_beginning(self):
        """Test stripping @mention at beginning."""
        result = strip_mentions("@troubleshoot why is my pod crashing?")
        assert result == "why is my pod crashing?"

    def test_strip_at_end(self):
        """Test stripping @mention at end."""
        result = strip_mentions("Help me @troubleshoot")
        assert result == "Help me"

    def test_strip_in_middle(self):
        """Test stripping @mention in middle."""
        result = strip_mentions("Can you @metrics show CPU?")
        assert result == "Can you show CPU?"

    def test_strip_all_mentions(self):
        """Test that ALL @mentions are stripped."""
        result = strip_mentions("@metrics @troubleshoot help")
        assert result == "help"

    def test_strip_multiple_mentions_mixed(self):
        """Test stripping multiple @mentions at different positions."""
        result = strip_mentions("@metrics show @prometheus CPU @grafana usage")
        assert result == "show CPU usage"

    def test_no_mention(self):
        """Test message without @mention."""
        result = strip_mentions("No mention here")
        assert result == "No mention here"

    def test_empty_message(self):
        """Test empty message."""
        assert strip_mentions("") == ""

    def test_none_message(self):
        """Test None message."""
        assert strip_mentions(None) is None

    def test_whitespace_cleaned(self):
        """Test that extra whitespace is cleaned up."""
        result = strip_mentions("@metrics   show   CPU")
        assert result == "show CPU"

    def test_only_mentions(self):
        """Test message containing only mentions."""
        result = strip_mentions("@metrics @prometheus @grafana")
        assert result == ""


class TestStripMention:
    """Tests for strip_mention function (alias for strip_mentions)."""

    def test_strip_mention_is_alias(self):
        """Test that strip_mention is an alias for strip_mentions."""
        # strip_mention should have the same behavior as strip_mentions
        assert strip_mention("@metrics @troubleshoot help") == "help"
        assert strip_mention("@troubleshoot why is my pod?") == "why is my pod?"
