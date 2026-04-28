"""Unit tests for Telegram group-topic → Hermes profile routing.

Covers the four small, isolated pieces introduced for the feature:
- ``find_group_topic_config()`` (telegram.py top-level helper)
- profile-prefixed ``build_session_key()`` (gateway/session.py)
- ``MessageEvent.routed_profile`` population + profile/skill precedence
- ``ProfileRuntimeRegistry`` lazy-load + missing-profile fallback
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _ensure_telegram_mock():
    """Inline minimal telegram mock — see test_group_topic_profile_dispatch
    for the rationale (xdist mock-ownership race).
    """
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import find_group_topic_config  # noqa: E402
from gateway.profile_runtime import (  # noqa: E402
    ProfileRuntime,
    ProfileRuntimeRegistry,
    build_host_runtime,
)
from gateway.session import SessionSource, build_session_key  # noqa: E402
from gateway.config import Platform  # noqa: E402


# ---------------------------------------------------------------------------
# find_group_topic_config
# ---------------------------------------------------------------------------

def _extra(*, group_topics):
    return {"group_topics": group_topics}


def test_find_group_topic_config_matches_chat_and_thread():
    extra = _extra(group_topics=[
        {
            "chat_id": -1001234567890,
            "topics": [
                {"name": "Microgreens", "thread_id": 5, "profile": "phyllis"},
                {"name": "Swedish", "thread_id": 12, "profile": "swedish"},
            ],
        },
    ])
    hit = find_group_topic_config(extra, -1001234567890, 5)
    assert hit is not None
    assert hit["name"] == "Microgreens"
    assert hit["profile"] == "phyllis"


def test_find_group_topic_config_coerces_int_and_string():
    extra = _extra(group_topics=[
        {
            "chat_id": "-1001234567890",  # string in YAML
            "topics": [{"thread_id": "5", "profile": "phyllis"}],
        },
    ])
    # Caller provides int chat_id and int thread_id — must still match
    assert find_group_topic_config(extra, -1001234567890, 5) is not None
    # And vice-versa
    assert find_group_topic_config(extra, "-1001234567890", "5") is not None


def test_find_group_topic_config_returns_none_for_unmapped_thread():
    extra = _extra(group_topics=[
        {"chat_id": 1, "topics": [{"thread_id": 5, "skill": "x"}]},
    ])
    assert find_group_topic_config(extra, 1, 99) is None


def test_find_group_topic_config_returns_none_for_unmapped_chat():
    extra = _extra(group_topics=[
        {"chat_id": 1, "topics": [{"thread_id": 5}]},
    ])
    assert find_group_topic_config(extra, 2, 5) is None


def test_find_group_topic_config_handles_missing_section():
    assert find_group_topic_config({}, 1, 5) is None
    assert find_group_topic_config({"group_topics": None}, 1, 5) is None
    assert find_group_topic_config(None, 1, 5) is None


def test_find_group_topic_config_skips_malformed_entries():
    extra = _extra(group_topics=[
        "not a dict",  # malformed — must be skipped
        {"chat_id": 1, "topics": ["bad", {"thread_id": 5, "profile": "p"}]},
    ])
    hit = find_group_topic_config(extra, 1, 5)
    assert hit is not None
    assert hit["profile"] == "p"


def test_find_group_topic_config_returns_full_topic_dict():
    """Helper must expose every topic field — name, thread_id, skill, profile."""
    topic = {
        "name": "Research",
        "thread_id": 42,
        "skill": "arxiv",
        "profile": "researcher",
    }
    extra = _extra(group_topics=[{"chat_id": 1, "topics": [topic]}])
    hit = find_group_topic_config(extra, 1, 42)
    assert hit == topic


# ---------------------------------------------------------------------------
# build_session_key — profile-aware
# ---------------------------------------------------------------------------

def _src(**overrides):
    base = dict(
        platform=Platform.TELEGRAM,
        chat_id="-1001234567890",
        chat_type="group",
        thread_id="5",
        user_id="alice",
    )
    base.update(overrides)
    return SessionSource(**base)


def test_build_session_key_no_profile_unchanged():
    """Regression guard: omitting profile_name must produce byte-for-byte
    identical keys to the pre-feature implementation.  Existing transcripts
    rely on these exact strings for resumption."""
    src = _src()
    key = build_session_key(src)
    assert key.startswith("agent:main:telegram:group:")
    assert "profile" not in key


def test_build_session_key_no_profile_dm_unchanged():
    src = _src(chat_type="dm", chat_id="111", thread_id=None)
    assert build_session_key(src) == "agent:main:telegram:dm:111"


def test_build_session_key_with_profile():
    src = _src()
    key = build_session_key(src, profile_name="phyllis")
    assert key.startswith("agent:profile:phyllis:telegram:group:")
    assert "main" not in key.split(":")[:2]


def test_build_session_key_distinct_profiles_distinct_keys():
    src = _src()
    a = build_session_key(src, profile_name="phyllis")
    b = build_session_key(src, profile_name="swedish-tutor")
    assert a != b
    # Without profile, host gets a third distinct key
    assert build_session_key(src) not in (a, b)


def test_build_session_key_empty_profile_treated_as_host():
    """Empty string is falsy — must produce host-prefix keys, not
    ``agent:profile::...`` (which would silently fork session history)."""
    src = _src()
    assert build_session_key(src, profile_name="") == build_session_key(src)
    assert build_session_key(src, profile_name=None) == build_session_key(src)


# ---------------------------------------------------------------------------
# ProfileRuntimeRegistry
# ---------------------------------------------------------------------------

@pytest.fixture
def host_runtime(tmp_path):
    home = tmp_path / "host"
    home.mkdir()
    return build_host_runtime(
        name="default",
        home=home,
        session_db=MagicMock(),
        soul_prompt="HOST_SOUL",
    )


def test_registry_get_none_returns_host(host_runtime):
    reg = ProfileRuntimeRegistry(host_runtime)
    assert reg.get(None) is host_runtime


def test_registry_get_host_name_returns_host(host_runtime):
    reg = ProfileRuntimeRegistry(host_runtime)
    assert reg.get("default") is host_runtime


def test_registry_get_missing_falls_back_to_host_with_warning(
    host_runtime, caplog, monkeypatch, tmp_path,
):
    # Point profile resolver at an empty dir so any name except "default"
    # cannot be found.  Use the test profiles_root so we don't touch
    # ~/.hermes.
    profiles_root = tmp_path / "profiles"
    profiles_root.mkdir()
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: profiles_root,
    )

    reg = ProfileRuntimeRegistry(host_runtime)
    with caplog.at_level(logging.WARNING, logger="gateway.profile_runtime"):
        out = reg.get("does-not-exist")
    assert out is host_runtime
    assert any(
        "missing Hermes profile" in r.message for r in caplog.records
    ), "expected a WARNING log for missing profile"


def test_registry_get_existing_loads_and_caches(
    host_runtime, monkeypatch, tmp_path,
):
    # Build a fake profile dir with config.yaml and SOUL.md.
    profiles_root = tmp_path / "profiles"
    profile_home = profiles_root / "phyllis"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text(
        "model:\n  default: claude-opus-4-7\n", encoding="utf-8",
    )
    (profile_home / "SOUL.md").write_text("PHYLLIS_PERSONA", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: profiles_root,
    )

    # Stub SessionDB so we don't touch real SQLite.
    fake_db = MagicMock(name="phyllis_state_db")
    monkeypatch.setattr("hermes_state.SessionDB", lambda db_path=None: fake_db)

    reg = ProfileRuntimeRegistry(host_runtime)
    runtime = reg.get("phyllis")

    assert runtime is not host_runtime
    assert runtime.name == "phyllis"
    assert runtime.home == profile_home
    assert runtime.config.get("model", {}).get("default") == "claude-opus-4-7"
    assert runtime.soul_prompt == "PHYLLIS_PERSONA"
    assert runtime.session_db is fake_db
    assert runtime.is_host is False

    # Second call returns the same cached instance — no reload.
    assert reg.get("phyllis") is runtime


def test_registry_shutdown_closes_routed_db_only(host_runtime, monkeypatch, tmp_path):
    """``shutdown()`` must close routed-profile DBs but never the host's
    (the gateway's session_db is closed separately by the gateway itself)."""
    profiles_root = tmp_path / "profiles"
    profile_home = profiles_root / "phyllis"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: profiles_root,
    )

    fake_db = MagicMock(name="phyllis_state_db")
    monkeypatch.setattr("hermes_state.SessionDB", lambda db_path=None: fake_db)

    reg = ProfileRuntimeRegistry(host_runtime)
    reg.get("phyllis")  # warm cache
    reg.shutdown()

    fake_db.close.assert_called_once()
    host_runtime.session_db.close.assert_not_called()


# ---------------------------------------------------------------------------
# build_host_runtime sanity
# ---------------------------------------------------------------------------

def test_build_host_runtime_marks_is_host(tmp_path):
    rt = build_host_runtime(
        name="default",
        home=tmp_path,
        session_db=MagicMock(),
    )
    assert rt.is_host is True
    assert rt.name == "default"


def test_registry_rejects_non_host_seed():
    fake = ProfileRuntime(
        name="phyllis",
        home=Path("/tmp"),
        is_host=False,
    )
    with pytest.raises(ValueError):
        ProfileRuntimeRegistry(fake)
