"""Persistent identities, network membership, devices, and run history."""

import hashlib
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .config import ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id), expires REAL NOT NULL);
CREATE TABLE IF NOT EXISTS networks (id TEXT PRIMARY KEY, name TEXT NOT NULL, password_hash TEXT NOT NULL, owner_id TEXT NOT NULL REFERENCES users(id), created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS memberships (network_id TEXT NOT NULL REFERENCES networks(id), user_id TEXT NOT NULL REFERENCES users(id), PRIMARY KEY(network_id,user_id));
CREATE TABLE IF NOT EXISTS nodes (id TEXT PRIMARY KEY, network_id TEXT NOT NULL REFERENCES networks(id), user_id TEXT NOT NULL REFERENCES users(id), device_key TEXT NOT NULL, label TEXT NOT NULL, mode TEXT NOT NULL DEFAULT 'running', capabilities TEXT NOT NULL DEFAULT '{}', last_seen REAL NOT NULL, completed INTEGER NOT NULL DEFAULT 0, received INTEGER NOT NULL DEFAULT 0, sent INTEGER NOT NULL DEFAULT 0, UNIQUE(network_id,user_id,device_key));
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, network_id TEXT NOT NULL REFERENCES networks(id), user_id TEXT NOT NULL REFERENCES users(id), name TEXT NOT NULL, jobs TEXT NOT NULL, created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS nodes_network ON nodes(network_id);
CREATE INDEX IF NOT EXISTS runs_network ON runs(network_id,created);
CREATE TABLE IF NOT EXISTS run_inputs (run_id TEXT PRIMARY KEY REFERENCES runs(id), payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS api_keys (id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL, network_id TEXT NOT NULL REFERENCES networks(id), user_id TEXT NOT NULL REFERENCES users(id), name TEXT NOT NULL, permission TEXT NOT NULL, created REAL NOT NULL, expires REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS task_timers (id TEXT PRIMARY KEY, network_id TEXT NOT NULL REFERENCES networks(id), user_id TEXT NOT NULL REFERENCES users(id), run_id TEXT NOT NULL REFERENCES runs(id), name TEXT NOT NULL, interval_seconds INTEGER NOT NULL, next_run REAL NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, last_run_id TEXT, last_fired REAL, last_error TEXT NOT NULL DEFAULT '', created REAL NOT NULL);
CREATE INDEX IF NOT EXISTS timers_due ON task_timers(enabled,next_run);
"""


@contextmanager
def connection():
    path = Path(os.getenv("HIVE_DB_PATH", str(ROOT / "data/hive.sqlite3")))
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    try:
        yield db
        db.commit()
    finally:
        db.close()


def query(sql, params=(), *, one=False):
    with connection() as db:
        rows = db.execute(sql, params)
        return (
            (dict(row) if (row := rows.fetchone()) else None)
            if one
            else [dict(row) for row in rows.fetchall()]
        )


def execute(sql, params=()):
    with connection() as db:
        db.execute(sql, params)


def save_run(network_id, user_id, name, jobs, payload):
    run_id = uuid4().hex
    with connection() as conn:
        conn.execute('INSERT INTO runs VALUES(?,?,?,?,?,?)',
                     (run_id, network_id, user_id, name, json.dumps(jobs), time.time()))
        conn.execute('INSERT INTO run_inputs VALUES(?,?)', (run_id, json.dumps(payload)))
    return run_id


def upsert_user(user):
    execute(
        "INSERT INTO users VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,email=excluded.email,updated=excluded.updated",
        (user.id, user.name, user.email, time.time()),
    )


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.scrypt(
        password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1
    ).hex()
    return f"scrypt${salt}${digest}"


def password_matches(password, stored):
    try:
        _, salt, _ = stored.split("$")
        return secrets.compare_digest(password_hash(password, salt), stored)
    except (ValueError, TypeError):
        return False


def session_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def save_session(token, user, expires):
    upsert_user(user)
    execute("DELETE FROM sessions WHERE expires <= ?", (time.time(),))
    execute(
        "INSERT INTO sessions VALUES(?,?,?)", (session_hash(token), user.id, expires)
    )


def read_session(token):
    return (
        query(
            "SELECT u.id,u.name,u.email,s.expires FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=? AND s.expires>?",
            (session_hash(token), time.time()),
            one=True,
        )
        if token
        else None
    )


def remove_session(token):
    if token:
        execute("DELETE FROM sessions WHERE token_hash=?", (session_hash(token),))


def member(network_id, user_id):
    return query(
        "SELECT n.id,n.name,n.owner_id,n.created FROM networks n JOIN memberships m ON n.id=m.network_id WHERE n.id=? AND m.user_id=?",
        (network_id, user_id),
        one=True,
    )


def enroll(network_id, user, device_key, label):
    node_id = uuid4().hex
    with connection() as db:
        db.execute(
            "INSERT OR IGNORE INTO nodes(id,network_id,user_id,device_key,label,last_seen) VALUES(?,?,?,?,?,?)",
            (node_id, network_id, user.id, device_key, label, time.time()),
        )
        return dict(
            db.execute(
                "SELECT * FROM nodes WHERE network_id=? AND user_id=? AND device_key=?",
                (network_id, user.id, device_key),
            ).fetchone()
        )


def save_node(worker, *, received=0, sent=0, completed=0):
    if worker.node_id:
        execute(
            "UPDATE nodes SET capabilities=?,last_seen=?,received=received+?,sent=sent+?,completed=completed+? WHERE id=?",
            (
                json.dumps(worker.capabilities.model_dump()),
                time.time(),
                received,
                sent,
                completed,
                worker.node_id,
            ),
        )
