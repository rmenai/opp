"""Testing settings related to environment variables."""

import os

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests() -> None:
    """Override os environments for tests."""
    os.environ["DEBUG"] = "False"
