"""Test the health endpoint."""

from http import HTTPStatus

import httpx


def test_healthz(client: httpx.Client) -> None:
    """Test /healthz."""
    resp = client.get("/healthz")
    assert resp.status_code == HTTPStatus.OK
    assert resp.json()["status"] == "ok"
