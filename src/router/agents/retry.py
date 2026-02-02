"""Retry logic with exponential backoff for agent requests.

This module provides retry configuration and utilities for retrying
failed agent requests with exponential backoff.
"""

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of attempts (including initial).
        base_delay_ms: Base delay in milliseconds for exponential backoff.
        max_delay_ms: Maximum delay in milliseconds.
    """

    max_attempts: int = 3
    base_delay_ms: int = 500
    max_delay_ms: int = 5000

    def get_delay_ms(self, attempt: int) -> int:
        """Calculate delay for a given attempt using exponential backoff.

        Args:
            attempt: The current attempt number (0-indexed).

        Returns:
            Delay in milliseconds before the next retry.
        """
        if attempt <= 0:
            return 0
        delay = self.base_delay_ms * (2 ** (attempt - 1))
        return min(delay, self.max_delay_ms)

    async def wait_before_retry(self, attempt: int) -> None:
        """Wait before retrying based on the attempt number.

        Args:
            attempt: The current attempt number (0-indexed).
        """
        delay_ms = self.get_delay_ms(attempt)
        if delay_ms > 0:
            logger.debug("Waiting %dms before retry attempt %d", delay_ms, attempt + 1)
            await asyncio.sleep(delay_ms / 1000.0)


@dataclass
class RetryResult:
    """Result of a retry operation.

    Attributes:
        success: Whether the operation succeeded.
        attempts: Number of attempts made.
        last_error: The last error encountered (if any).
    """

    success: bool
    attempts: int
    last_error: Exception | None = None


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error should trigger a retry.

    Retryable errors include:
    - Connection errors (timeouts, connection refused)
    - 5xx server errors
    - Temporary unavailability

    Non-retryable errors include:
    - 4xx client errors (except 429 Too Many Requests)
    - Validation errors
    - Authentication errors

    Args:
        error: The exception to check.

    Returns:
        True if the error is retryable.
    """
    error_str = str(error).lower()

    # Check for connection/timeout errors
    if any(
        term in error_str
        for term in [
            "timeout",
            "timed out",
            "connection",
            "connect",
            "unavailable",
            "network",
        ]
    ):
        return True

    # Check for HTTP status codes
    if "429" in error_str:  # Too Many Requests
        return True
    if "5" in error_str and any(code in error_str for code in ["500", "502", "503", "504"]):
        return True

    # Check for specific error attributes
    if hasattr(error, "status_code"):
        status_code = getattr(error, "status_code", None)
        if status_code is not None:
            # Retry on 429 or 5xx
            if status_code == 429 or 500 <= status_code < 600:
                return True
            # Don't retry on 4xx (except 429)
            if 400 <= status_code < 500:
                return False

    return False
