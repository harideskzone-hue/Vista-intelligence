"""
SIH26187 — Authentication Utilities
=======================================
Provides:
  • bcrypt password hashing / verification
  • CSRF token generation and validation
  • Session age check (expiry)
  • Credential load / save (data/auth.json)
"""

import os
import json
import time
import secrets
import hashlib
import bcrypt    # type: ignore

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR    = os.path.join(_HERE, "..", "data")
_AUTH_FILE   = os.path.join(_DATA_DIR, "auth.json")

os.makedirs(_DATA_DIR, exist_ok=True)

# ── Session / CSRF config ──────────────────────────────────────────────────────
SESSION_MAX_AGE = 8 * 60 * 60   # seconds — 8 hours idle timeout
CSRF_TOKEN_LEN  = 32

# ── Password helpers ────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Return a bcrypt hash string for *plain*."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed* bcrypt string."""
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── Credential storage ──────────────────────────────────────────────────────────

def is_setup_complete() -> bool:
    """Returns True when an admin password has been set."""
    return os.path.exists(_AUTH_FILE)


def load_credentials() -> dict:
    """Load auth.json; returns {} if missing."""
    if not os.path.exists(_AUTH_FILE):
        return {}
    try:
        with open(_AUTH_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_credentials(password_hash: str) -> None:
    """Persist the bcrypt password hash to auth.json."""
    creds = {"password_hash": password_hash, "created_at": time.time()}
    with open(_AUTH_FILE, "w") as f:
        json.dump(creds, f, indent=2)


# ── Session helpers ─────────────────────────────────────────────────────────────

def is_session_valid(session: dict) -> bool:
    """
    Returns True if session contains a valid auth marker AND
    was set within SESSION_MAX_AGE seconds.
    """
    if not session.get("authenticated"):
        return False
    logged_in_at = session.get("logged_in_at", 0)
    return (time.time() - logged_in_at) < SESSION_MAX_AGE


def create_session(session: dict) -> None:
    """Mark session as authenticated and record timestamp."""
    session["authenticated"]  = True
    session["logged_in_at"]   = time.time()
    session["csrf_token"]     = secrets.token_hex(CSRF_TOKEN_LEN)


def clear_session(session: dict) -> None:
    """Remove all auth keys from session."""
    for key in ("authenticated", "logged_in_at", "csrf_token"):
        session.pop(key, None)


# ── CSRF helpers ────────────────────────────────────────────────────────────────

def generate_csrf_token(session: dict) -> str:
    """Return (and store) a CSRF token for the current session."""
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(CSRF_TOKEN_LEN)
    return session["csrf_token"]


def validate_csrf(session: dict, token: str | None) -> bool:
    """Constant-time comparison of submitted token vs session token."""
    expected = session.get("csrf_token", "")
    if not token or not expected:
        return False
    return secrets.compare_digest(expected, token)
