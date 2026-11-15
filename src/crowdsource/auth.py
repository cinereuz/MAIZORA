"""Password-based admin authentication and in-memory token store."""

import os
import secrets
import threading
import time


class AdminAuth:
    """Password-based admin authentication backed by `ADMIN_PASSWORD` in `.env`."""

    TOKEN_TTL = 8 * 3600

    def __init__(self):
        self.password = os.environ.get("ADMIN_PASSWORD", "").strip()
        self._tokens = {}
        self._lock = threading.Lock()
        if not self.password:
            print(
                "[Warning] ADMIN_PASSWORD is not set -- admin login is DISABLED. "
                "Create a '.env' file (see '.env.example') containing ADMIN_PASSWORD=..."
            )

    @property
    def enabled(self):
        return bool(self.password)

    def login(self, password):
        if not self.enabled:
            return None
        if not secrets.compare_digest(str(password or ""), self.password):
            return None
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._purge()
            self._tokens[token] = time.time() + self.TOKEN_TTL
        return token

    def logout(self, token):
        with self._lock:
            self._tokens.pop(token, None)

    def is_valid(self, token):
        if not token:
            return False
        with self._lock:
            exp = self._tokens.get(token)
            if exp is None:
                return False
            if exp < time.time():
                self._tokens.pop(token, None)
                return False
            return True

    def _purge(self):
        now = time.time()
        for t in [t for t, e in self._tokens.items() if e < now]:
            self._tokens.pop(t, None)
