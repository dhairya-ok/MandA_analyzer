"""backend package initializer.

This file makes `backend` a Python package so imports like
`backend.api.main` and relative imports inside `backend` work.
"""

__all__ = ["api", "db", "models"]
