"""Integration tests for group_topics → MessageEvent.routed_profile.

End-to-end through ``TelegramAdapter._build_message_event``: a config
with two routed profiles + one skill-only topic must produce three
distinct ``MessageEvent``s, each carrying the right ``routed_profile``
and ``auto_skill`` values.  Profile takes precedence over skill (with
WARNING).  Backward compat: topics without ``profile`` are unchanged.
"""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _ensure_telegram_mock():
    """Inline minimal telegram mock — mirrors ``test_dm_topics._ensure_telegram_mock``.

    Uses ``setdefault`` so it never fights another file's already-installed
    mock for ownership.  Without this, xdist workers that collected
    ``test_dm_topics.py`` first would have a different ``ChatType`` MagicMock
    bound into ``gateway.platforms.telegram`` than the one this test file
    later resolves via ``from telegram.constants import ChatType``, breaking
    the ``chat.type in (ChatType.SUPERGROUP, ...)`` equality check.
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

from gateway.config import PlatformConfig  # noqa: E402
from gateway.platforms.base import MessageType  # noqa: E402
from gateway.platforms.telegram import TelegramAdapter  # noqa: E402

# Same trick as test_dm_topics.py — production code does
# ``from telegram.constants import ChatType``, which resolves through the
# mock chain.  We must use the same ChatType the production code sees.
from telegram.constants import ChatType as _ChatType  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirrored from test_dm_topics for consistency)
# ---------------------------------------------------------------------------


def _make_adapter(group_topics_config):
    extra = {"group_topics": group_topics_config}
    config = PlatformConfig(enabled=True, token="***", extra=extra)
    return TelegramAdapter(config)


def _make_supergroup_message(chat_id, thread_id, text="hi"):
    chat = SimpleNamespace(id=chat_id, type=_ChatType.SUPERGROUP, title="Test Group")
    chat.full_name = "Test Group"
    chat.is_forum = True
    user = SimpleNamespace(id=42, full_name="Alice")
    return SimpleNamespace(
        chat=chat,
        from_user=user,
        text=text,
        message_thread_id=thread_id,
        message_id=1,
        reply_to_message=None,
        date=None,
    )


# ---------------------------------------------------------------------------
# Three-topic dispatch table
# ---------------------------------------------------------------------------

GROUP_ID = -1001234567890

THREE_TOPIC_CONFIG = [
    {
        "chat_id": GROUP_ID,
        "topics": [
            # Topic A: routed to profile, no skill
            {"name": "Microgreens", "thread_id": 5, "profile": "phyllis"},
            # Topic B: routed to a different profile
            {"name": "Swedish", "thread_id": 12, "profile": "swedish-tutor"},
            # Topic C: legacy skill-only binding (must keep working)
            {"name": "Research", "thread_id": 42, "skill": "arxiv"},
        ],
    },
]


def test_topic_a_routes_to_phyllis():
    adapter = _make_adapter(THREE_TOPIC_CONFIG)
    msg = _make_supergroup_message(GROUP_ID, 5, text="how do I grow basil")
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.routed_profile == "phyllis"
    assert event.auto_skill is None  # profile-only topic
    assert event.source.chat_topic == "Microgreens"
    assert event.source.thread_id == "5"


def test_topic_b_routes_to_swedish_tutor():
    adapter = _make_adapter(THREE_TOPIC_CONFIG)
    msg = _make_supergroup_message(GROUP_ID, 12, text="hej")
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.routed_profile == "swedish-tutor"
    assert event.auto_skill is None
    assert event.source.chat_topic == "Swedish"


def test_topic_c_keeps_legacy_skill_binding_unchanged():
    """Backward-compat: a topic without ``profile`` must continue to set
    ``auto_skill`` exactly as before, with ``routed_profile=None``."""
    adapter = _make_adapter(THREE_TOPIC_CONFIG)
    msg = _make_supergroup_message(GROUP_ID, 42, text="find me a paper")
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.routed_profile is None
    assert event.auto_skill == "arxiv"
    assert event.source.chat_topic == "Research"


def test_unmapped_topic_has_no_routing():
    """A thread_id that's not in config produces a vanilla event."""
    adapter = _make_adapter(THREE_TOPIC_CONFIG)
    msg = _make_supergroup_message(GROUP_ID, 999)
    event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.routed_profile is None
    assert event.auto_skill is None


# ---------------------------------------------------------------------------
# Profile + skill precedence
# ---------------------------------------------------------------------------


def test_profile_wins_over_skill_with_warning(caplog):
    """When both ``profile`` and ``skill`` are configured, ``profile`` wins
    and a WARNING is emitted naming the topic so an operator can find it."""
    adapter = _make_adapter([
        {
            "chat_id": GROUP_ID,
            "topics": [
                {
                    "name": "Mixed",
                    "thread_id": 7,
                    "profile": "phyllis",
                    "skill": "arxiv",
                },
            ],
        },
    ])
    msg = _make_supergroup_message(GROUP_ID, 7)
    with caplog.at_level(logging.WARNING):
        event = adapter._build_message_event(msg, MessageType.TEXT)

    assert event.routed_profile == "phyllis"
    assert event.auto_skill is None  # skill MUST be ignored
    # Operator-facing log must mention the offending coordinates
    assert any(
        "profile takes precedence" in r.message
        and "phyllis" in r.message
        and "arxiv" in r.message
        for r in caplog.records
    ), f"expected precedence warning, got {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Cross-traffic isolation: routed turns and host turns produce distinct
# session keys, so the gateway agent cache will not blend them.
# ---------------------------------------------------------------------------


def test_routed_and_host_session_keys_are_distinct():
    """Two messages in the *same* supergroup, one routed and one not, must
    produce different session keys so the agent cache treats them as
    independent conversations."""
    from gateway.session import build_session_key

    adapter = _make_adapter(THREE_TOPIC_CONFIG)
    routed_msg = _make_supergroup_message(GROUP_ID, 5)
    host_msg = _make_supergroup_message(GROUP_ID, 999)  # unmapped → host

    routed_event = adapter._build_message_event(routed_msg, MessageType.TEXT)
    host_event = adapter._build_message_event(host_msg, MessageType.TEXT)

    routed_key = build_session_key(
        routed_event.source, profile_name=routed_event.routed_profile,
    )
    host_key = build_session_key(host_event.source)

    assert routed_key.startswith("agent:profile:phyllis:")
    assert host_key.startswith("agent:main:")
    assert routed_key != host_key
