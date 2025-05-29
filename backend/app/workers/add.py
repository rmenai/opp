"""Example task file."""

from app.worker import app


@app.task
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
