"""Test session endpoints."""

import logging
import uuid
from http import HTTPStatus

import httpx
import pytest

log = logging.getLogger(__name__)


@pytest.fixture
def created_session(client: httpx.Client, auth_headers: dict[str, str]) -> dict:
    """Fixture to create a new session, yield it and clean it afterwards."""
    content = {"type": "file", "metadata": {"from": "pytest_session"}}
    resp = client.post("/sessions", headers=auth_headers, json=content)
    resp.raise_for_status()
    session_data = resp.json()

    assert session_data["type"] == "file"
    assert session_data["status"] == "pending"

    yield session_data

    log.debug("Cleaning up session: %s", session_data.get("session_id"))
    cleanup_resp = client.delete(f"/sessions/{session_data['session_id']}", headers=auth_headers)
    if cleanup_resp.status_code not in [HTTPStatus.OK, HTTPStatus.NOT_FOUND]:
        log.warning("Problem cleaning up session: %s", session_data.get("session_id"))


def test_create_session(client: httpx.Client, auth_headers: dict[str, str], created_session: dict) -> None:
    """Test creating a new session."""


def test_list_sessions(client: httpx.Client, auth_headers: dict[str, str], created_session: dict) -> None:
    """Test querying all sessions."""
    resp = client.get("/sessions", headers=auth_headers)
    assert resp.status_code == HTTPStatus.OK
    sessions = resp.json()
    assert isinstance(sessions, list)
    assert any(s["session_id"] == created_session["session_id"] for s in sessions)


def test_get_specific_session(client: httpx.Client, auth_headers: dict[str, str], created_session: dict) -> None:
    """Test verifying a specific session by its ID."""
    session_id = created_session["session_id"]
    resp = client.get(f"/sessions/{session_id}", headers=auth_headers)
    assert resp.status_code == HTTPStatus.OK
    assert resp.json() == created_session


def test_close_session(client: httpx.Client, auth_headers: dict[str, str], created_session: dict) -> None:
    """Test closing an active session."""
    session_id = created_session["session_id"]

    resp_close = client.post(f"/sessions/{session_id}/close", headers=auth_headers)
    assert resp_close.status_code == HTTPStatus.OK

    # Verify the session is now closed
    resp = client.get(f"/sessions/{session_id}", headers=auth_headers)
    assert resp.status_code == HTTPStatus.OK
    session = resp.json()
    assert session["status"] == "closed"
    assert session["session_id"] == session_id

    # Verify the closed session does not appear in the active list
    resp_list = client.get("/sessions", headers=auth_headers)
    assert resp_list.status_code == HTTPStatus.OK
    sessions = resp_list.json()
    assert not any(sess["session_id"] == session_id for sess in sessions)


def test_delete_session(client: httpx.Client, auth_headers: dict[str, str], created_session: dict) -> None:
    """Test deleting a session."""
    session_id = created_session["session_id"]

    resp = client.delete(f"/sessions/{session_id}", headers=auth_headers)
    assert resp.status_code == HTTPStatus.OK

    # Verify the session is no longer accessible
    resp_get = client.get(f"/sessions/{session_id}", headers=auth_headers)
    assert resp_get.status_code == HTTPStatus.NOT_FOUND


def test_get_non_existent_session(client: httpx.Client, auth_headers: dict[str, str]) -> None:
    """Test querying a fake/non-existent session."""
    fake_session_id = uuid.uuid4()
    resp = client.get(f"/sessions/{fake_session_id}", headers=auth_headers)
    assert resp.status_code == HTTPStatus.NOT_FOUND


def test_delete_non_existent_session(client: httpx.Client, auth_headers: dict[str, str]) -> None:
    """Test deleting a fake/non-existent session."""
    fake_session_id = uuid.uuid4()
    resp = client.delete(f"/sessions/{fake_session_id}", headers=auth_headers)
    assert resp.status_code == HTTPStatus.NOT_FOUND


def test_close_non_existent_session(client: httpx.Client, auth_headers: dict[str, str]) -> None:
    """Test closing a fake/non-existent session."""
    fake_session_id = uuid.uuid4()
    resp = client.post(f"/sessions/{fake_session_id}/close", headers=auth_headers)
    assert resp.status_code == HTTPStatus.NOT_FOUND
