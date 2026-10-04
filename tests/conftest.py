"""Keep authentication/database tests isolated from local user data."""

import pytest


@pytest.fixture(autouse=True)
def isolated_database(tmp_path, monkeypatch):
    monkeypatch.setenv("HIVE_DB_PATH", str(tmp_path / "hive.sqlite3"))
    monkeypatch.setenv("HIVE_ALLOW_LEGACY_POOL", "false")
    from backend import auth

    auth.sessions.clear()
    yield
    auth.sessions.clear()
