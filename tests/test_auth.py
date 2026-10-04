"""Exercise real token verification with local signing keys; no Google network calls."""
import time
import sqlite3

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient
from google.auth import crypt, jwt
from google.auth.exceptions import TransportError

from backend import auth

CLIENT_ID = 'test-client.apps.googleusercontent.com'


@pytest.fixture
def identity(monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signer = crypt.RSASigner.from_string(private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()), key_id='test-key')
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    # Only the certificates transport is replaced. Google's verifier checks the
    # actual JWT signature, audience, issuer, and timestamps.
    monkeypatch.setattr(auth.id_token, '_fetch_certs', lambda *args: {'test-key': public_key})
    monkeypatch.setenv('GOOGLE_CLIENT_ID', CLIENT_ID)
    monkeypatch.setenv('COOKIE_SECURE', 'false')
    auth.sessions.clear()

    def token(**changes):
        now = int(time.time())
        claims = dict(sub='google-user-123', name='Alex Chen', email='alex@example.com',
                      email_verified=True, iss='https://accounts.google.com',
                      aud=CLIENT_ID, iat=now - 5, exp=now + 3600)
        claims.update(changes)
        return jwt.encode(signer, claims).decode()

    app = FastAPI()
    app.include_router(auth.router)
    with TestClient(app) as client:
        yield client, token
    auth.sessions.clear()


def csrf_headers(client):
    config = client.get('/auth/config')
    assert config.headers['cache-control'] == 'no-store'
    return {'X-CSRF-Token': config.json()['csrf_token']}


def test_login_refresh_rotation_and_logout(identity):
    client, token = identity
    assert client.get('/auth/me').json() == {'user': None}
    headers = csrf_headers(client)
    signed_in = client.post('/auth/google', json={'credential': token()}, headers=headers)
    assert signed_in.status_code == 200
    user = signed_in.json()['user']
    assert user == {'id': 'google-user-123', 'name': 'Alex Chen', 'email': 'alex@example.com'}
    cookie = signed_in.headers['set-cookie']
    assert 'HttpOnly' in cookie and 'SameSite=lax' in cookie and 'Path=/' in cookie
    assert client.get('/auth/me').json()['user'] == user
    old_session = client.cookies.get(auth.SESSION_COOKIE)
    assert client.post('/auth/google', json={'credential': token()}, headers=headers).status_code == 200
    assert old_session not in auth.sessions
    session = client.cookies.get(auth.SESSION_COOKIE)
    assert client.post('/auth/logout', headers=headers).status_code == 200
    assert client.get('/auth/me').json()['user'] is None
    assert session not in auth.sessions
    assert auth.SESSION_COOKIE not in client.cookies


@pytest.mark.parametrize('claims', [
    {'aud': 'another-app'}, {'iss': 'https://attacker.example'},
    {'exp': 1}, {'email_verified': False}, {'sub': ''},
])
def test_reject_invalid_identity(identity, claims):
    client, token = identity
    response = client.post('/auth/google', json={'credential': token(**claims)},
                           headers=csrf_headers(client))
    assert response.status_code == 401
    assert not auth.sessions
    assert client.get('/auth/me').json()['user'] is None


def test_reject_forged_signature(identity):
    client, token = identity
    header, payload, signature = token().split('.')
    forged = f'{header}.{payload}.{"A" if signature[0] != "A" else "B"}{signature[1:]}'
    assert client.post('/auth/google', json={'credential': forged},
                       headers=csrf_headers(client)).status_code == 401
    assert not auth.sessions


@pytest.mark.parametrize('path', ['/auth/google', '/auth/guest', '/auth/logout'])
def test_csrf_required(identity, path):
    client, token = identity
    payload = {'credential': token()} if path.endswith('google') else None
    assert client.post(path, json=payload).status_code == 403
    csrf_headers(client)
    assert client.post(path, json=payload, headers={'X-CSRF-Token': 'wrong'}).status_code == 403


def test_session_expiry(identity, monkeypatch):
    client, token = identity
    assert client.post('/auth/google', json={'credential': token()},
                       headers=csrf_headers(client)).status_code == 200
    now = time.time()
    monkeypatch.setattr(auth.time, 'time', lambda: now + auth.SESSION_SECONDS + 1)
    assert client.get('/auth/me').json()['user'] is None
    assert not auth.sessions


def test_missing_config_and_google_unavailable(identity, monkeypatch):
    client, token = identity
    headers = csrf_headers(client)
    credential = token()
    monkeypatch.setenv('GOOGLE_CLIENT_ID', '')
    assert client.get('/auth/config').json()['client_id'] == ''
    assert client.post('/auth/google', json={'credential': credential}, headers=headers).status_code == 503
    monkeypatch.setenv('GOOGLE_CLIENT_ID', CLIENT_ID)

    def unavailable(*args):
        raise TransportError('Google unavailable')

    monkeypatch.setattr(auth.id_token, '_fetch_certs', unavailable)
    assert client.post('/auth/google', json={'credential': credential}, headers=headers).status_code == 503
    assert not auth.sessions


@pytest.mark.parametrize('failure', [PermissionError('storage is not writable'), sqlite3.OperationalError('unable to open database file')])
def test_login_storage_failure_has_actionable_response_and_no_memory_session(identity, monkeypatch, failure):
    client, token = identity
    def unavailable(*args):
        raise failure
    monkeypatch.setattr(auth.database, 'save_session', unavailable)
    response = client.post('/auth/google', json={'credential': token()}, headers=csrf_headers(client))
    assert response.status_code == 503
    assert 'database storage' in response.json()['detail']
    assert not auth.sessions
    assert auth.SESSION_COOKIE not in response.cookies


def test_guest_without_google_refresh_restart_and_logout(identity, monkeypatch):
    client, _ = identity
    monkeypatch.setenv('GOOGLE_CLIENT_ID', '')
    headers = csrf_headers(client)
    response = client.post('/auth/guest', json={}, headers=headers)
    assert response.status_code == 200
    user = response.json()['user']
    assert user['id'].startswith('guest:')
    assert user['name'] == 'Guest' and user['email'] == ''
    assert 'HttpOnly' in response.headers['set-cookie']
    assert response.headers['cache-control'] == 'no-store'
    token = client.cookies.get(auth.SESSION_COOKIE)
    auth.sessions.clear()  # Recover the same guest from SQLite after a restart.
    assert client.get('/auth/me').json()['user'] == user
    assert client.post('/auth/guest', json={}, headers=headers).json()['user'] == user
    assert client.cookies.get(auth.SESSION_COOKIE) == token
    assert client.post('/auth/logout', headers=headers).status_code == 200
    assert client.get('/auth/me').json()['user'] is None
    replacement = client.post('/auth/guest', json={}, headers=headers).json()['user']
    assert replacement['id'] != user['id']


def test_guest_button_does_not_replace_google_account(identity):
    client, token = identity
    headers = csrf_headers(client)
    user = client.post('/auth/google', json={'credential': token()}, headers=headers).json()['user']
    assert client.post('/auth/guest', json={}, headers=headers).json()['user'] == user


def test_guest_session_expires(identity, monkeypatch):
    client, _ = identity
    client.post('/auth/guest', json={}, headers=csrf_headers(client))
    now = time.time()
    monkeypatch.setattr(auth.time, 'time', lambda: now + auth.SESSION_SECONDS + 1)
    assert client.get('/auth/me').json()['user'] is None
