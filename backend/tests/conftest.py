"""A generated example file that is supposed to be modified."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.__main__ import app


@pytest.fixture(scope="module")
def client() -> Generator:
    """Return the test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def random_product() -> dict[str, str]:
    """Return a product example."""
    return {
        "name": "Test Product",
        "price": "80",
    }
