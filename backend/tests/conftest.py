"""A generated example file that is supposed to be modified."""

import time
from http import HTTPStatus

import httpx
import pytest

from tests.settings import BASE_API_URL


def pytest_sessionstart(session: pytest.Session) -> None:
    """
    Before pytest collects any tests, keep polling /healthz for up to 60s.

    If we see a 200, return and let pytest continue; otherwise abort.
    """
    health_url = "/healthz"

    timeout = 60.0
    interval = 1.0
    start = time.time()

    client = httpx.Client(base_url=BASE_API_URL, timeout=3.0)
    last_exc = None

    while time.time() - start < timeout:
        try:
            resp = client.get(health_url)
            if resp.status_code == HTTPStatus.OK:
                client.close()
                return  # healthy → proceed with collecting/running tests
        except httpx.RequestError as e:
            last_exc = e
        time.sleep(interval)

    client.close()
    msg = (
        f"/healthz did not return 200 within {timeout:g}s\n"
        f"Last exception: {last_exc!r}\n"
        f"Last status: {resp.status_code if 'resp' in locals() else 'n/a'}\n"
    )
    pytest.exit(msg, returncode=1)


@pytest.fixture(scope="module")
def client() -> httpx.Client:
    """Return a real HTTP client pointing to the live API."""
    with httpx.Client(base_url=BASE_API_URL, timeout=5.0) as client:
        yield client
