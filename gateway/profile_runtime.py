"""Per-profile runtime registry for Telegram group-topic profile routing.

When a Telegram group forum topic is configured with ``profile: <name>`` in
``platforms.telegram.extra.group_topics``, the gateway dispatches that
topic's messages into the named Hermes profile's runtime: separate
``state.db`` handle, separate ``config.yaml`` (model, provider, toolsets),
and the profile's ``SOUL.md`` as the system prompt.

The gateway process never mutates ``os.environ["HERMES_HOME"]`` per
message — many Hermes modules cache that env var at import time. Instead,
each non-host profile's resolved values are threaded into the ``AIAgent``
constructor explicitly.

Per-profile API credentials (`.env`) are intentionally out of scope: the
spec called out model / skills / memory / SOUL as the per-topic isolation
boundary; credentials remain shared with the host. Folding profile `.env`
files into per-turn credential resolution is a documented follow-up.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProfileRuntime:
    """Resolved runtime for a single Hermes profile.

    Frozen so the registry can hand the same instance to every routed turn
    without worrying about callers accidentally mutating cached state.

    ``is_host=True`` for the profile the gateway process was launched with.
    For host profiles, ``session_db`` is the gateway-owned shared handle and
    must NOT be closed by ``ProfileRuntimeRegistry.shutdown()``.
    """
    name: str
    home: Path
    config: Dict[str, Any] = field(default_factory=dict)
    soul_prompt: Optional[str] = None
    session_db: Any = None  # hermes_state.SessionDB; typed Any to avoid hard import
    is_host: bool = False


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning(
                "Profile config %s did not parse as a YAML mapping; ignoring.",
                path,
            )
            return {}
        return data
    except Exception as exc:  # noqa: BLE001 — best-effort load
        logger.warning("Failed to read profile config %s: %s", path, exc)
        return {}


def read_profile_text(path: Path) -> Optional[str]:
    """Read a UTF-8 text file from a profile dir, returning ``None`` on miss
    or read error. Used for ``SOUL.md`` and (transitively) any future
    profile-local prompt files. Public so the gateway can call it for the
    host profile's SOUL without reaching into a private helper."""
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read profile file %s: %s", path, exc)
        return None


class ProfileRuntimeRegistry:
    """Lazily-loaded, thread-safe cache of ``ProfileRuntime`` instances.

    The host profile (the one the gateway was launched under) is registered
    at construction time and always returned for ``profile_name=None`` or
    when the requested name matches the host.  Other profiles are loaded on
    first access and cached for the lifetime of the gateway.

    On a missing profile directory, ``get()`` logs a WARNING and returns the
    host runtime as a fallback — this matches the user's stated preference
    that misconfiguration must not break Telegram traffic.
    """

    def __init__(self, host_runtime: ProfileRuntime):
        if not host_runtime.is_host:
            raise ValueError("host_runtime must have is_host=True")
        self._host = host_runtime
        self._lock = threading.Lock()
        self._cache: Dict[str, ProfileRuntime] = {host_runtime.name: host_runtime}

    @property
    def host_name(self) -> str:
        return self._host.name

    @property
    def host_runtime(self) -> ProfileRuntime:
        return self._host

    def get(self, profile_name: Optional[str]) -> ProfileRuntime:
        """Return the runtime for ``profile_name``, falling back to host on miss."""
        if not profile_name or profile_name == self._host.name:
            return self._host

        with self._lock:
            cached = self._cache.get(profile_name)
            if cached is not None:
                return cached

        runtime = self._load(profile_name)
        with self._lock:
            # Another thread may have raced us; honor whichever entry won.
            existing = self._cache.get(profile_name)
            if existing is not None:
                # Close the loser's DB so we don't leak a handle.
                if runtime is not existing and not runtime.is_host:
                    self._safe_close(runtime)
                return existing
            self._cache[profile_name] = runtime
        return runtime

    def _load(self, profile_name: str) -> ProfileRuntime:
        """Resolve ``profile_name`` from disk, or return host on miss.

        Performed outside the registry lock so a slow disk read doesn't
        block other lookups.
        """
        try:
            from hermes_cli.profiles import get_profile_dir
        except ImportError:
            logger.error(
                "hermes_cli.profiles unavailable; cannot route to profile %s — "
                "falling back to host profile %s.",
                profile_name, self._host.name,
            )
            return self._host

        try:
            home = get_profile_dir(profile_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to resolve profile dir for %s: %s — falling back to host %s.",
                profile_name, exc, self._host.name,
            )
            return self._host

        if not home.is_dir():
            logger.warning(
                "Telegram topic routes to missing Hermes profile %s "
                "(expected dir %s); falling back to host profile %s.",
                profile_name, home, self._host.name,
            )
            return self._host

        config = _read_yaml(home / "config.yaml")
        soul = read_profile_text(home / "SOUL.md")

        try:
            from hermes_state import SessionDB
            session_db = SessionDB(db_path=home / "state.db")
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to open state.db for profile %s at %s: %s — "
                "falling back to host profile %s.",
                profile_name, home, exc, self._host.name,
            )
            return self._host

        logger.info(
            "Loaded ProfileRuntime for %s (home=%s, has_soul=%s)",
            profile_name, home, soul is not None,
        )

        return ProfileRuntime(
            name=profile_name,
            home=home,
            config=config,
            soul_prompt=soul,
            session_db=session_db,
            is_host=False,
        )

    @staticmethod
    def _safe_close(runtime: ProfileRuntime) -> None:
        db = runtime.session_db
        if db is None:
            return
        try:
            db.close()
        except Exception:  # noqa: BLE001
            pass

    def shutdown(self) -> None:
        """Close every non-host ``SessionDb`` handle.

        Intended for the gateway's shutdown path so transcripts get a clean
        WAL checkpoint.  Idempotent.
        """
        with self._lock:
            entries = list(self._cache.items())
            self._cache = {self._host.name: self._host}
        for name, runtime in entries:
            if runtime.is_host:
                continue
            self._safe_close(runtime)
            logger.debug("Closed ProfileRuntime session_db for %s", name)


def build_host_runtime(
    *,
    name: str,
    home: Path,
    session_db: Any,
    config: Optional[Dict[str, Any]] = None,
    soul_prompt: Optional[str] = None,
) -> ProfileRuntime:
    """Helper used by ``GatewayRunner.__init__`` to wrap the host profile.

    ``session_db`` here is the gateway-owned shared handle; the registry
    will treat ``is_host=True`` runtimes as non-closable so we don't tear
    down the gateway's DB during shutdown twice.
    """
    return ProfileRuntime(
        name=name,
        home=home,
        config=config or {},
        soul_prompt=soul_prompt,
        session_db=session_db,
        is_host=True,
    )
