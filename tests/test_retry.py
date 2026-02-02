"""Test retry logic and exponential backoff."""

import asyncio

import pytest

from router.agents.retry import (
    RetryConfig,
    RetryResult,
    is_retryable_error,
)


class TestRetryConfig:
    """Test RetryConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay_ms == 500
        assert config.max_delay_ms == 5000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay_ms=100,
            max_delay_ms=2000,
        )

        assert config.max_attempts == 5
        assert config.base_delay_ms == 100
        assert config.max_delay_ms == 2000

    def test_get_delay_ms_first_attempt(self):
        """Test no delay for first attempt."""
        config = RetryConfig(base_delay_ms=100)

        assert config.get_delay_ms(0) == 0

    def test_get_delay_ms_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(base_delay_ms=100, max_delay_ms=10000)

        # attempt 1: 100 * 2^0 = 100
        assert config.get_delay_ms(1) == 100

        # attempt 2: 100 * 2^1 = 200
        assert config.get_delay_ms(2) == 200

        # attempt 3: 100 * 2^2 = 400
        assert config.get_delay_ms(3) == 400

        # attempt 4: 100 * 2^3 = 800
        assert config.get_delay_ms(4) == 800

    def test_get_delay_ms_capped_at_max(self):
        """Test delay is capped at max_delay_ms."""
        config = RetryConfig(base_delay_ms=1000, max_delay_ms=2000)

        # attempt 1: 1000 * 2^0 = 1000
        assert config.get_delay_ms(1) == 1000

        # attempt 2: 1000 * 2^1 = 2000 (at max)
        assert config.get_delay_ms(2) == 2000

        # attempt 3: 1000 * 2^2 = 4000, but capped at 2000
        assert config.get_delay_ms(3) == 2000

        # attempt 10: would be huge, but capped at 2000
        assert config.get_delay_ms(10) == 2000

    @pytest.mark.asyncio
    async def test_wait_before_retry_no_wait_first_attempt(self):
        """Test no wait for first attempt."""
        config = RetryConfig(base_delay_ms=1000)

        start = asyncio.get_event_loop().time()
        await config.wait_before_retry(0)
        elapsed = asyncio.get_event_loop().time() - start

        # Should be nearly instant
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_wait_before_retry_waits(self):
        """Test wait_before_retry actually waits."""
        config = RetryConfig(base_delay_ms=50, max_delay_ms=100)

        start = asyncio.get_event_loop().time()
        await config.wait_before_retry(1)
        elapsed = asyncio.get_event_loop().time() - start

        # Should wait approximately 50ms
        assert 0.04 < elapsed < 0.15


class TestRetryResult:
    """Test RetryResult dataclass."""

    def test_success_result(self):
        """Test successful retry result."""
        result = RetryResult(success=True, attempts=1)

        assert result.success is True
        assert result.attempts == 1
        assert result.last_error is None

    def test_failure_result(self):
        """Test failed retry result."""
        error = ValueError("test error")
        result = RetryResult(success=False, attempts=3, last_error=error)

        assert result.success is False
        assert result.attempts == 3
        assert result.last_error == error


class TestIsRetryableError:
    """Test is_retryable_error function."""

    def test_timeout_error_is_retryable(self):
        """Test that timeout errors are retryable."""
        error = Exception("Request timed out")
        assert is_retryable_error(error) is True

    def test_connection_error_is_retryable(self):
        """Test that connection errors are retryable."""
        assert is_retryable_error(Exception("Connection refused")) is True
        assert is_retryable_error(Exception("Connect timeout")) is True
        assert is_retryable_error(Exception("Network unreachable")) is True

    def test_server_errors_are_retryable(self):
        """Test that 5xx server errors are retryable."""
        assert is_retryable_error(Exception("HTTP 500 Internal Server Error")) is True
        assert is_retryable_error(Exception("HTTP 502 Bad Gateway")) is True
        assert is_retryable_error(Exception("HTTP 503 Service Unavailable")) is True
        assert is_retryable_error(Exception("HTTP 504 Gateway Timeout")) is True

    def test_too_many_requests_is_retryable(self):
        """Test that 429 Too Many Requests is retryable."""
        error = Exception("HTTP 429 Too Many Requests")
        assert is_retryable_error(error) is True

    def test_client_errors_not_retryable(self):
        """Test that 4xx client errors (except 429) are not retryable."""
        # 400 Bad Request - not retryable
        error_400 = Exception("HTTP 400 Bad Request")
        assert is_retryable_error(error_400) is False

        # 401 Unauthorized - not retryable
        error_401 = Exception("HTTP 401 Unauthorized")
        assert is_retryable_error(error_401) is False

        # 404 Not Found - not retryable
        error_404 = Exception("HTTP 404 Not Found")
        assert is_retryable_error(error_404) is False

    def test_status_code_attribute(self):
        """Test errors with status_code attribute."""

        class HTTPError(Exception):
            def __init__(self, status_code):
                super().__init__(f"HTTP {status_code}")
                self.status_code = status_code

        # 500 is retryable
        assert is_retryable_error(HTTPError(500)) is True

        # 502 is retryable
        assert is_retryable_error(HTTPError(502)) is True

        # 429 is retryable
        assert is_retryable_error(HTTPError(429)) is True

        # 400 is not retryable
        assert is_retryable_error(HTTPError(400)) is False

        # 401 is not retryable
        assert is_retryable_error(HTTPError(401)) is False

    def test_unavailable_error_is_retryable(self):
        """Test that unavailable errors are retryable."""
        error = Exception("Service unavailable")
        assert is_retryable_error(error) is True

    def test_generic_error_not_retryable(self):
        """Test that generic errors are not retryable by default."""
        error = Exception("Some random error")
        assert is_retryable_error(error) is False
