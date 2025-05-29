"""Test auth endpoints."""

from http import HTTPStatus

import httpx

from tests.settings import EMAIL, PASSWORD


def test_register_existing_username(client: httpx.Client, auth_headers: dict[str, str]) -> None:  # noqa: ARG001
    """Try registering the same username twice."""
    payload = {"username": EMAIL, "password": PASSWORD}
    resp = client.post("/auth/register", data=payload)
    assert resp.status_code == HTTPStatus.BAD_REQUEST


def test_login_bad_credentials(client: httpx.Client) -> None:
    """Test logging in with fake credentials."""
    payload = {"username": "doesnotexist@nop.com", "password": "nopw"}
    resp = client.post("/auth/login", data=payload)
    assert resp.status_code == HTTPStatus.UNAUTHORIZED


def test_login_good_credentials(client: httpx.Client) -> None:
    """Test logging in with fake credentials."""
    payload = {"username": EMAIL, "password": PASSWORD}
    resp = client.post("/auth/login", data=payload)
    assert resp.status_code == HTTPStatus.OK


def test_protected_route_requires_auth(client: httpx.Client) -> None:
    """Test a protected route without auth."""
    resp = client.get("/me")
    assert resp.status_code == HTTPStatus.UNAUTHORIZED


def test_get_profile(client: httpx.Client, auth_headers: dict[str, str]) -> None:
    """Test a protected route with auth."""
    resp = client.get("/me", headers=auth_headers)
    assert resp.status_code == HTTPStatus.OK
    assert resp.json()["email"] == EMAIL


def test_update_profile(client: httpx.Client, auth_headers: dict[str, str]) -> None:
    """Test a protected route with auth."""
    content_test = {"language": "fr", "keyboard_layout": "AZERTY"}
    content_old = {"language": "en", "keyboard_layout": "QWERTY"}

    resp = client.put("/me", headers=auth_headers, json=content_test)
    assert resp.status_code == HTTPStatus.OK
    assert resp.json() == {
        "email": "test@test.com",
        "language": "fr",
        "keyboard_layout": "AZERTY",
    }

    resp = client.get("/me", headers=auth_headers)
    assert resp.status_code == HTTPStatus.OK
    assert resp.json() == {
        "email": "test@test.com",
        "language": "fr",
        "keyboard_layout": "AZERTY",
    }

    resp = client.put("/me", headers=auth_headers, json=content_old)
    assert resp.status_code == HTTPStatus.OK
    assert resp.json() == {
        "email": "test@test.com",
        "language": "en",
        "keyboard_layout": "QWERTY",
    }


def test_update_profile_bad_data(client: httpx.Client, auth_headers: dict[str, str]) -> None:
    """Test a protected route with auth."""
    content = {"languages": "fr", "keyboard_layout": "AZERTY"}
    resp = client.put("/me", headers=auth_headers, json=content)
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_CONTENT

    content = {"languages": 1, "keyboard_layout": "AZERTY"}
    resp = client.put("/me", headers=auth_headers, json=content)
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_CONTENT
