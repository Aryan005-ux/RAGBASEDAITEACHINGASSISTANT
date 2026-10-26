"""User accounts and login sessions, stored in SQLite (DATA_DIR/app.db).

Passwords are hashed with PBKDF2-SHA256. Sessions are random tokens kept in an
HttpOnly cookie; only a SHA-256 of each token is stored, so a leaked database
can't be used to log in.
"""
import os
import re
import time
import hmac
import secrets
import hashlib
import sqlite3
import threading
from collections import defaultdict, deque
from fastapi import Request, HTTPException
from app.config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "app.db")
SESSION_COOKIE = "session"
SESSION_DAYS = 30
PBKDF2_ITERATIONS = 600_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Failed-login throttling: at most MAX_FAILURES per email (and per IP) in FAILURE_WINDOW seconds
MAX_FAILURES = 10
FAILURE_WINDOW = 15 * 60
_failures = defaultdict(deque)
_failures_lock = threading.Lock()


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    with connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created REAL NOT NULL,
                expires REAL NOT NULL
            );
        """)


# ---------- Passwords ----------
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(candidate.hex(), digest)
    except ValueError:
        return False


def public_user(row) -> dict:
    return {"id": row["id"], "email": row["email"], "name": row["name"]}


# ---------- Users ----------
def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def create_user(email: str, name: str, password: str) -> dict:
    email, name = normalize_email(email), (name or "").strip()
    if not EMAIL_RE.match(email):
        raise ValueError("Please enter a valid email address.")
    if not name:
        raise ValueError("Please enter your name.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    user_id = secrets.token_hex(8)
    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO users (id, email, name, password_hash, created) VALUES (?, ?, ?, ?, ?)",
                (user_id, email, name[:80], hash_password(password), time.time()),
            )
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email already exists. Try logging in.")
    return {"id": user_id, "email": email, "name": name[:80]}


def user_count() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def _too_many_failures(*keys) -> bool:
    now = time.time()
    with _failures_lock:
        for key in keys:
            q = _failures[key]
            while q and q[0] < now - FAILURE_WINDOW:
                q.popleft()
            if len(q) >= MAX_FAILURES:
                return True
    return False


def _record_failure(*keys):
    with _failures_lock:
        for key in keys:
            _failures[key].append(time.time())


def authenticate(email: str, password: str, ip: str) -> dict:
    email = normalize_email(email)
    if _too_many_failures(f"email:{email}", f"ip:{ip}"):
        raise PermissionError("Too many failed attempts. Please wait 15 minutes and try again.")
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        _record_failure(f"email:{email}", f"ip:{ip}")
        raise ValueError("Incorrect email or password.")
    return public_user(row)


# ---------- Sessions ----------
def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE expires < ?", (now,))
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, created, expires) VALUES (?, ?, ?, ?)",
            (_token_hash(token), user_id, now, now + SESSION_DAYS * 86400),
        )
    return token


def delete_session(token: str):
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def user_for_token(token: str):
    if not token:
        return None
    with connect() as conn:
        row = conn.execute(
            "SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id "
            "WHERE sessions.token_hash = ? AND sessions.expires > ?",
            (_token_hash(token), time.time()),
        ).fetchone()
    return public_user(row) if row else None


# ---------- FastAPI dependencies ----------
def optional_user(request: Request):
    return user_for_token(request.cookies.get(SESSION_COOKIE))


def current_user(request: Request) -> dict:
    user = optional_user(request)
    if not user:
        raise HTTPException(401, "Please log in.")
    return user


init_db()
