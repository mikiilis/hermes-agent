"""Tests for Telegram message reactions tied to processing lifecycle hooks."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    # ReactionTypeEmoji is used by the hooks; make it a simple callable wrapper
    telegram_mod.ReactionTypeEmoji = lambda emoji: SimpleNamespace(emoji=emoji)

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def _make_adapter(reactions: dict | None = None) -> TelegramAdapter:
    config = PlatformConfig(enabled=True, token="test-token")
    if reactions is not None:
        config.extra["reactions"] = reactions
    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = config
    adapter._bot = MagicMock()
    adapter._bot.set_message_reaction = AsyncMock()
    return adapter


def _make_event(message_id: str = "42") -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="123",
            chat_type="dm",
        ),
        message_id=message_id,
    )


@pytest.mark.asyncio
async def test_on_processing_start_sets_default_acknowledge_reaction():
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_start(event)

    adapter._bot.set_message_reaction.assert_awaited_once()
    call_kwargs = adapter._bot.set_message_reaction.await_args.kwargs
    assert call_kwargs["chat_id"] == 123
    assert call_kwargs["message_id"] == 42
    assert call_kwargs["reaction"][0].emoji == "👀"


@pytest.mark.asyncio
async def test_on_processing_complete_success_sets_default_reaction():
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, success=True)

    adapter._bot.set_message_reaction.assert_awaited_once()
    assert adapter._bot.set_message_reaction.await_args.kwargs["reaction"][0].emoji == "👍"


@pytest.mark.asyncio
async def test_on_processing_complete_failure_sets_default_reaction():
    adapter = _make_adapter()
    event = _make_event()

    await adapter.on_processing_complete(event, success=False)

    adapter._bot.set_message_reaction.assert_awaited_once()
    assert adapter._bot.set_message_reaction.await_args.kwargs["reaction"][0].emoji == "👎"


@pytest.mark.asyncio
async def test_custom_reactions_from_config():
    adapter = _make_adapter(reactions={"acknowledge": "😎", "success": "🔥", "failure": "💔"})
    event = _make_event()

    await adapter.on_processing_start(event)
    assert adapter._bot.set_message_reaction.await_args.kwargs["reaction"][0].emoji == "😎"

    adapter._bot.set_message_reaction.reset_mock()
    await adapter.on_processing_complete(event, success=True)
    assert adapter._bot.set_message_reaction.await_args.kwargs["reaction"][0].emoji == "🔥"

    adapter._bot.set_message_reaction.reset_mock()
    await adapter.on_processing_complete(event, success=False)
    assert adapter._bot.set_message_reaction.await_args.kwargs["reaction"][0].emoji == "💔"


@pytest.mark.asyncio
async def test_reaction_failure_does_not_propagate():
    adapter = _make_adapter()
    adapter._bot.set_message_reaction = AsyncMock(side_effect=RuntimeError("no perms"))
    event = _make_event()

    # Neither hook should raise even when the Telegram API call fails
    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, success=True)
    await adapter.on_processing_complete(event, success=False)


@pytest.mark.asyncio
async def test_no_bot_skips_reaction():
    adapter = _make_adapter()
    adapter._bot = None
    event = _make_event()

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, success=True)


@pytest.mark.asyncio
async def test_no_message_id_skips_reaction():
    adapter = _make_adapter()
    event = _make_event(message_id=None)

    await adapter.on_processing_start(event)
    await adapter.on_processing_complete(event, success=True)

    adapter._bot.set_message_reaction.assert_not_awaited()
