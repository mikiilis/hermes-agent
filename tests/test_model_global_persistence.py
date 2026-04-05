"""Regression tests for /model --global config persistence.

Bug: both the CLI REPL and gateway /model handlers were writing to
`model.name` instead of `model.default`, while all startup code reads
`model.default`. This made --global persistence a silent no-op.

CLI tests use source inspection (importing cli at module level fails due
to optional prompt_toolkit dependency), gateway tests are functional.
"""

import inspect
import yaml
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_cli.model_switch import ModelSwitchResult


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_switch_result(model="claude-sonnet-4-6", provider="openrouter"):
    return ModelSwitchResult(
        success=True,
        new_model=model,
        target_provider=provider,
        provider_changed=True,
    )


def _make_gateway_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._session_model_overrides = {}
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


def _make_message_event(text: str):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user1",
        chat_id="chat1",
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


# ---------------------------------------------------------------------------
# CLI source inspection tests
# (cli.py can't be imported at module level in tests due to optional deps)
# ---------------------------------------------------------------------------

class TestCliModelGlobalPersistenceSource:

    def test_handler_uses_model_default_not_model_name(self):
        """_handle_model_switch must call save_config_value with 'model.default'."""
        source = inspect.getsource(gateway_run.GatewayRunner._handle_model_command)
        # Verify via the gateway handler as a proxy for the shared fix pattern;
        # the CLI source check is done by reading the raw file below.
        import pathlib
        cli_source = pathlib.Path("cli.py").read_text(encoding="utf-8")
        assert 'save_config_value("model.default"' in cli_source, (
            "CLI _handle_model_switch must use 'model.default', not 'model.name'"
        )
        assert 'save_config_value("model.name"' not in cli_source, (
            "Found stale 'model.name' key in cli.py — regression of the --global bug"
        )


# ---------------------------------------------------------------------------
# Gateway functional tests
# ---------------------------------------------------------------------------

class TestGatewayModelGlobalPersistence:

    @pytest.fixture
    def hermes_env(self, tmp_path, monkeypatch):
        """Isolated Hermes home wired into both gateway_run and hermes_constants."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        return hermes_home

    @pytest.mark.asyncio
    async def test_global_writes_model_default_not_name(self, hermes_env):
        """Gateway /model --global must write model.default, not model.name."""
        config_path = hermes_env / "config.yaml"
        config_path.write_text(yaml.dump({
            "model": {"default": "claude-opus-4-6", "provider": "openrouter"},
        }))

        runner = _make_gateway_runner()
        event = _make_message_event("/model sonnet --global")

        with patch("hermes_cli.model_switch.switch_model", return_value=_make_switch_result()), \
             patch("hermes_cli.model_switch.list_authenticated_providers", return_value=[]):
            await runner._handle_model_command(event)

        saved = yaml.safe_load(config_path.read_text())
        model = saved["model"]
        assert model.get("default") == "claude-sonnet-4-6", (
            f"Expected model.default to be set, got: {model}"
        )
        assert "name" not in model, (
            f"model.name should not exist in config after --global, got: {model}"
        )

    @pytest.mark.asyncio
    async def test_global_does_not_clobber_other_config_keys(self, hermes_env):
        """--global write must preserve unrelated config entries."""
        config_path = hermes_env / "config.yaml"
        config_path.write_text(yaml.dump({
            "model": {"default": "claude-opus-4-6", "provider": "openrouter"},
            "display": {"skin": "ares"},
        }))

        runner = _make_gateway_runner()
        event = _make_message_event("/model sonnet --global")

        with patch("hermes_cli.model_switch.switch_model", return_value=_make_switch_result()), \
             patch("hermes_cli.model_switch.list_authenticated_providers", return_value=[]):
            await runner._handle_model_command(event)

        saved = yaml.safe_load(config_path.read_text())
        assert saved["display"]["skin"] == "ares"

    @pytest.mark.asyncio
    async def test_session_only_does_not_write_config(self, hermes_env):
        """Without --global, config.yaml must not be modified."""
        config_path = hermes_env / "config.yaml"
        original = yaml.dump({
            "model": {"default": "claude-opus-4-6", "provider": "openrouter"},
        })
        config_path.write_text(original)

        runner = _make_gateway_runner()
        event = _make_message_event("/model sonnet")

        with patch("hermes_cli.model_switch.switch_model", return_value=_make_switch_result()), \
             patch("hermes_cli.model_switch.list_authenticated_providers", return_value=[]):
            await runner._handle_model_command(event)

        assert config_path.read_text() == original, (
            "Session-only switch must not modify config.yaml"
        )
