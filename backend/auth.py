"""Google and guest sign-in with short-lived, SQLite-backed sessions."""
import os
import secrets
import time
from functools import partial
from threading import Lock

from fastapi import APIRouter, HTTPException, Request, Response
from google.auth.exceptions import GoogleAuthError, TransportError
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token
from pydantic import BaseModel, ConfigDict, Field
from . import database

router = APIRouter(prefix='/auth', tags=['auth'])
SESSION_COOKIE = 'hive_session'
CSRF_COOKIE = 'hive_csrf'
SESSION_SECONDS = 3600
sessions = {}
sessions_lock = Lock()


class GoogleLogin(BaseModel):
    model_config = ConfigDict(extra='forbid')
    credential: str = Field(min_length=1, max_length=8192)


class User(BaseModel):
    id: str = Field(min_length=1)
    name: str
    email: str


def cookie_options():
    return dict(httponly=True, secure=os.getenv('COOKIE_SECURE', 'false').lower() == 'true',
                samesite='lax', path='/')


def prune_sessions():
    now = time.time()
    for key, (_, expires) in list(sessions.items()):
        if expires <= now:
            sessions.pop(key, None)


def require_csrf(request):
    cookie = request.cookies.get(CSRF_COOKIE, '')
    header = request.headers.get('X-CSRF-Token', '')
    if not cookie or not secrets.compare_digest(cookie.encode(), header.encode()):
        raise HTTPException(403, 'Refresh the page and try again.')


@router.get('/config')
def config(request: Request, response: Response):
    response.headers['Cache-Control'] = 'no-store'
    csrf = request.cookies.get(CSRF_COOKIE) or secrets.token_urlsafe(32)
    response.set_cookie(CSRF_COOKIE, csrf, max_age=SESSION_SECONDS, **cookie_options())
    return {'client_id': os.getenv('GOOGLE_CLIENT_ID', '').strip(), 'csrf_token': csrf}


@router.post('/google')
def google_login(payload: GoogleLogin, request: Request, response: Response):
    require_csrf(request)
    client_id = os.getenv('GOOGLE_CLIENT_ID', '').strip()
    if not client_id:
        raise HTTPException(503, 'Google sign-in is not configured yet.')
    try:
        claims = id_token.verify_oauth2_token(
            payload.credential, partial(GoogleRequest(), timeout=10), client_id)
        if claims.get('email_verified') is not True:
            raise ValueError('Unverified email')
        user = User(id=claims.get('sub'), name=claims.get('name') or claims.get('email'),
                    email=claims.get('email'))
    except TransportError as exc:
        raise HTTPException(503, 'Google is temporarily unavailable. Please try again.') from exc
    except (ValueError, GoogleAuthError, KeyError) as exc:
        raise HTTPException(401, 'Google could not verify this account. Please try again.') from exc

    return start_session(user, request, response)


def start_session(user, request, response):
    """Issue the same protected session for Google accounts and guests."""
    with sessions_lock:
        prune_sessions()
        previous = request.cookies.get(SESSION_COOKIE)
        if len(sessions) >= 1024 and previous not in sessions:
            raise HTTPException(429, 'Sign-in is busy. Please try again later.')
        sessions.pop(previous, None)
        database.remove_session(previous)
        session = secrets.token_urlsafe(32)
        sessions[session] = (user, time.time() + SESSION_SECONDS)
        database.save_session(session, user, time.time() + SESSION_SECONDS)
    response.headers['Cache-Control'] = 'no-store'
    response.set_cookie(SESSION_COOKIE, session, max_age=SESSION_SECONDS, **cookie_options())
    return {'user': user}


@router.post('/guest')
def guest_login(request: Request, response: Response):
    require_csrf(request)
    response.headers['Cache-Control'] = 'no-store'
    existing = user_for_request(request)
    if existing:
        return {'user': existing}
    user = User(id=f'guest:{secrets.token_hex(16)}', name='Guest', email='')
    return start_session(user, request, response)


@router.get('/me')
def current_user(request: Request, response: Response):
    user = user_for_request(request)
    response.headers['Cache-Control'] = 'no-store'
    if not user:
        response.delete_cookie(SESSION_COOKIE, **cookie_options())
    return {'user': user}


def user_for_request(request):
    token = request.cookies.get(SESSION_COOKIE)
    with sessions_lock:
        prune_sessions()
        session = sessions.get(token)
    if session:
        return session[0]
    saved = database.read_session(token)
    return User(id=saved['id'], name=saved['name'], email=saved['email']) if saved else None


def require_user(request):
    user = user_for_request(request)
    if not user:
        raise HTTPException(401, 'Sign in to continue.')
    if request.scope['type'] == 'http' and request.method not in ('GET', 'HEAD', 'OPTIONS'):
        require_csrf(request)
    return user


@router.post('/logout')
def logout(request: Request, response: Response):
    require_csrf(request)
    with sessions_lock:
        sessions.pop(request.cookies.get(SESSION_COOKIE), None)
        database.remove_session(request.cookies.get(SESSION_COOKIE))
    response.headers['Cache-Control'] = 'no-store'
    response.delete_cookie(SESSION_COOKIE, **cookie_options())
    return {'user': None}
