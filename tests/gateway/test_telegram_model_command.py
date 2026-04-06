"""Tests for Telegram model selection pagination configuration."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from gateway.config import PlatformConfig, GatewayConfig, Platform, load_gateway_config

def _ensure_telegram_mock():
    """Mock the telegram package if it's not installed."""
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)

_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter

class TestTelegramPaginationConfig:
    """Tests for pagination configuration in TelegramAdapter."""

    def test_default_pagination_values(self):
        # No extra config provided
        config = PlatformConfig(enabled=True, token="test-token")
        adapter = TelegramAdapter(config)
        assert adapter._providers_per_page == 6
        assert adapter._models_per_page == 8

    def test_custom_pagination_values(self):
        # Custom config provided in extra
        config = PlatformConfig(
            enabled=True, 
            token="test-token",
            extra={
                "model_command": {
                    "providers_per_page": 10,
                    "models_per_page": 20
                }
            }
        )
        adapter = TelegramAdapter(config)
        assert adapter._providers_per_page == 10
        assert adapter._models_per_page == 20

    def test_invalid_pagination_values_fallback(self):
        # Invalid types should fallback to defaults (if int conversion fails) or use defaults if keys missing
        config = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={
                "model_command": "not-a-dict"
            }
        )
        adapter = TelegramAdapter(config)
        assert adapter._providers_per_page == 6
        assert adapter._models_per_page == 8

class TestTelegramConfigBridging:
    """Tests for bridging model_command from config.yaml to GatewayConfig."""

    def test_bridges_model_command_from_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        
        # Note: model_command is now at the top level of the telegram block in DEFAULT_CONFIG
        # and bridged explicitly in load_gateway_config
        config_path.write_text(
            "telegram:\n"
            "  model_command:\n"
            "    providers_per_page: 12\n"
            "    models_per_page: 15\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        
        # Mocking DEFAULT_CONFIG to include the telegram section if it wasn't there
        # (Though it is there now in the real code)
        config = load_gateway_config()

        assert Platform.TELEGRAM in config.platforms
        extra = config.platforms[Platform.TELEGRAM].extra
        assert "model_command" in extra
        assert extra["model_command"]["providers_per_page"] == 12
        assert extra["model_command"]["models_per_page"] == 15


class TestBuildProviderKeyboard:
    """Unit tests for TelegramAdapter._build_provider_keyboard layout."""

    def _make_adapter(self, providers_per_page=6):
        config = PlatformConfig(
            enabled=True,
            token="test-token",
            extra={"model_command": {"providers_per_page": providers_per_page, "models_per_page": 8}},
        )
        return TelegramAdapter(config)

    def _make_providers(self, count, current_slug=None):
        return [
            {"slug": f"p{i}", "name": f"Provider {i}", "is_current": (f"p{i}" == current_slug)}
            for i in range(count)
        ]

    def _get_all_callback_datas(self, rows):
        """Flatten all button callback_data values from a list of rows."""
        return [btn.callback_data for row in rows for btn in row]

    def _call_keyboard(self, adapter, providers, page, current_model="m", current_provider="p0"):
        """Call _build_provider_keyboard capturing rows via InlineKeyboardMarkup mock."""
        captured = {}

        class CapturingMarkup:
            def __init__(self, rows):
                captured["rows"] = rows

        # get_label is imported inline inside the method, so patch at its source module
        with patch("gateway.platforms.telegram.InlineKeyboardMarkup", side_effect=CapturingMarkup), \
             patch("gateway.platforms.telegram.InlineKeyboardButton", side_effect=lambda text, callback_data: MagicMock(text=text, callback_data=callback_data)), \
             patch("hermes_cli.providers.get_label", return_value="Provider X"):
            adapter._build_provider_keyboard(providers, page, current_model, current_provider)
        return captured.get("rows", [])

    def test_single_page_no_nav_row(self):
        adapter = self._make_adapter(providers_per_page=6)
        providers = self._make_providers(3)
        rows = self._call_keyboard(adapter, providers, 0)
        datas = self._get_all_callback_datas(rows)
        assert "mdl:providers:prev" not in datas
        assert "mdl:providers:next" not in datas

    def test_multi_page_first_page_next_only(self):
        adapter = self._make_adapter(providers_per_page=6)
        providers = self._make_providers(7)
        rows = self._call_keyboard(adapter, providers, page=0)
        datas = self._get_all_callback_datas(rows)
        assert "mdl:providers:next" in datas
        assert "mdl:providers:prev" not in datas

    def test_multi_page_last_page_prev_only(self):
        adapter = self._make_adapter(providers_per_page=6)
        providers = self._make_providers(7)
        rows = self._call_keyboard(adapter, providers, page=1)
        datas = self._get_all_callback_datas(rows)
        assert "mdl:providers:prev" in datas
        assert "mdl:providers:next" not in datas

    def test_middle_page_both_nav_buttons(self):
        adapter = self._make_adapter(providers_per_page=6)
        providers = self._make_providers(19)
        rows = self._call_keyboard(adapter, providers, page=1)
        datas = self._get_all_callback_datas(rows)
        assert "mdl:providers:prev" in datas
        assert "mdl:providers:next" in datas

    def test_current_provider_checkmark(self):
        adapter = self._make_adapter(providers_per_page=6)
        providers = self._make_providers(3, current_slug="p1")
        rows = self._call_keyboard(adapter, providers, page=0, current_provider="p1")
        # Flatten all button texts
        texts = [btn.text for row in rows for btn in row]
        assert any(t.endswith(" ✓") for t in texts), "Expected a ✓ marker on current provider"
        marked = [t for t in texts if t.endswith(" ✓")]
        assert len(marked) == 1

    def test_out_of_range_page_clamped(self):
        adapter = self._make_adapter(providers_per_page=6)
        providers = self._make_providers(3)
        # Should not raise; page 99 clamped to 0
        rows = self._call_keyboard(adapter, providers, page=99)
        assert rows is not None

    def test_footer_buttons_always_present(self):
        adapter = self._make_adapter(providers_per_page=6)
        providers = self._make_providers(3)
        rows = self._call_keyboard(adapter, providers, page=0)
        datas = self._get_all_callback_datas(rows)
        assert "mdl:manual" in datas
        assert "mdl:cancel" in datas
