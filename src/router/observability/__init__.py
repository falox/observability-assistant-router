"""Observability module for the router.

This module provides:
- StreamBuffer: Reassembles streaming message chunks into complete messages
- AuditLogger: Structured JSON audit logging for compliance and debugging
"""

from router.observability.audit import AuditLogger
from router.observability.buffer import StreamBuffer
from router.observability.models import (
    AuditEvent,
    AuditEventType,
    BufferedMessage,
    StreamFrame,
)

__all__ = [
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "BufferedMessage",
    "StreamBuffer",
    "StreamFrame",
]
