"""AG-UI protocol module for endpoint and client implementations.

Note: agui_router is imported lazily in main.py to avoid circular imports.
"""

from router.agui.models import ChatRequest

__all__ = ["ChatRequest", "agui_router"]


def get_agui_router():
    """Get the AG-UI router, deferring import to avoid circular imports."""
    from router.agui.endpoint import agui_router

    return agui_router
