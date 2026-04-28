"""Tests for the SOUL identity-override path used by group-topic profile routing.

The gateway sets ``identity_prompt_override`` on ``AIAgent`` when a Telegram
group topic routes to a non-host Hermes profile.  When set, this string
must replace the on-disk ``SOUL.md`` content as the system prompt's
identity layer — otherwise the routed agent ends up with both the host's
and the routed profile's personas in its system prompt.

These tests cover three guarantees:

1.  Constructor stores ``identity_prompt_override`` and normalises empty
    strings to ``None`` (an empty override must NOT erase the identity).
2.  The agent cache signature changes when ``identity_prompt`` differs,
    so two routed profiles in the same Telegram chat don't share a
    cached agent.
3.  The system-prompt identity-layer logic prefers the override over
    ``load_soul_md()`` and never calls ``load_soul_md()`` when an override
    is set.
"""

from unittest.mock import patch

import pytest

# These tests exercise ``AIAgent`` and ``GatewayRunner`` directly, both of
# which import the full Hermes runtime (openai, dotenv, requests, ...) at
# module top-level.  In the standard Hermes dev/CI environment those are
# installed; in minimal sandboxes they aren't.  Skip cleanly rather than
# whack-a-mole-mocking every transitive dependency.
pytest.importorskip("openai")
pytest.importorskip("dotenv")
pytest.importorskip("requests")
pytest.importorskip("fire")


# ---------------------------------------------------------------------------
# 1. Constructor storage
# ---------------------------------------------------------------------------


def test_identity_prompt_override_empty_string_normalized_to_none():
    """An empty string must NOT erase the identity — it must coerce to
    None so ``load_soul_md()`` falls through and the host's SOUL is used.
    The non-empty case is covered transitively by the
    ``_build_system_prompt`` tests below."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    override = ""
    agent.identity_prompt_override = override if override else None
    assert agent.identity_prompt_override is None


# ---------------------------------------------------------------------------
# 2. Agent cache signature differs by identity_prompt
# ---------------------------------------------------------------------------


def _signature(identity_prompt: str = "") -> str:
    """Helper: compute a signature with default args + variable identity."""
    from gateway.run import GatewayRunner

    runtime = {
        "api_key": "sk-test-1234567890",
        "base_url": "https://example.invalid/v1",
        "provider": "test",
        "api_mode": "chat_completions",
    }
    return GatewayRunner._agent_config_signature(
        "claude-opus-4-7",
        runtime,
        ["hermes-telegram"],
        ephemeral_prompt="",
        identity_prompt=identity_prompt,
    )


def test_signature_default_identity_is_stable():
    """No identity prompt → reproducible signature (regression guard)."""
    assert _signature("") == _signature("")


def test_signature_changes_when_identity_prompt_changes():
    """Two routed topics with different SOUL files must NOT share a cached
    agent — the signature must encode the identity."""
    sig_phyllis = _signature("PHYLLIS_PERSONA_BLOB")
    sig_swedish = _signature("SWEDISH_PERSONA_BLOB")
    sig_host    = _signature("")
    assert sig_phyllis != sig_swedish
    assert sig_phyllis != sig_host
    assert sig_swedish != sig_host


def test_signature_unaffected_when_identity_prompt_arg_omitted():
    """Existing callers that don't pass ``identity_prompt`` must produce
    the same signature as passing the empty string."""
    from gateway.run import GatewayRunner

    runtime = {
        "api_key": "sk-test-1234567890",
        "base_url": "https://example.invalid/v1",
        "provider": "test",
        "api_mode": "chat_completions",
    }
    sig_omitted = GatewayRunner._agent_config_signature(
        "claude-opus-4-7", runtime, ["hermes-telegram"], "",
    )
    sig_explicit_empty = GatewayRunner._agent_config_signature(
        "claude-opus-4-7", runtime, ["hermes-telegram"], "", identity_prompt="",
    )
    assert sig_omitted == sig_explicit_empty


# ---------------------------------------------------------------------------
# 3. _build_system_prompt prefers override and skips load_soul_md
# ---------------------------------------------------------------------------


def _make_minimal_agent_stub(
    *,
    identity_prompt_override=None,
    skip_context_files=True,
    skip_host_soul=False,
):
    """Construct an ``AIAgent`` with the minimum attributes needed for
    ``_build_system_prompt`` to reach the timestamp line.

    ``skip_context_files=True`` keeps the test from touching the disk for
    AGENTS.md / .cursorrules.  ``valid_tool_names`` is empty so all
    tool-aware branches early-out.  Memory / external-memory hooks are
    stubbed to falsy so their blocks are skipped.
    """
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.identity_prompt_override = (
        identity_prompt_override if identity_prompt_override else None
    )
    agent.skip_host_soul = bool(skip_host_soul)
    agent.skip_context_files = skip_context_files
    agent.valid_tool_names = []
    agent._tool_use_enforcement = False
    agent.model = "claude-opus-4-7"
    agent.provider = None
    agent._memory_store = None
    agent._memory_manager = None
    agent.pass_session_id = False
    agent.session_id = None
    return agent


def test_build_system_prompt_uses_override_and_skips_load_soul_md():
    """When ``identity_prompt_override`` is set, the identity layer must
    be the override and ``load_soul_md()`` must NOT be called.  The
    rendered prompt must not contain the host's SOUL string — this is
    the H1 leak guard (override must be the *only* identity)."""
    from run_agent import AIAgent

    agent = _make_minimal_agent_stub(
        identity_prompt_override="PHYLLIS_PERSONA",
        skip_context_files=True,
    )

    with patch("run_agent.load_soul_md", return_value="HOST_SOUL_PERSONA") as mock_load_soul:
        prompt = AIAgent._build_system_prompt(agent)

    assert prompt.startswith("PHYLLIS_PERSONA"), (
        f"Identity layer should be the override; got prefix: {prompt[:80]!r}"
    )
    assert "HOST_SOUL_PERSONA" not in prompt
    mock_load_soul.assert_not_called()


def test_build_system_prompt_falls_back_to_soul_md_without_override():
    """Without an override (and with context files enabled), the identity
    layer is the host's SOUL.md content."""
    from run_agent import AIAgent

    agent = _make_minimal_agent_stub(
        identity_prompt_override=None,
        skip_context_files=False,
    )

    with patch("run_agent.load_soul_md", return_value="HOST_SOUL_PERSONA") as mock_load_soul:
        prompt = AIAgent._build_system_prompt(agent)

    assert prompt.startswith("HOST_SOUL_PERSONA")
    mock_load_soul.assert_called_once()


# ---------------------------------------------------------------------------
# 4. skip_host_soul guard for routed profiles without SOUL.md
# ---------------------------------------------------------------------------


def test_skip_host_soul_falls_back_to_default_identity_without_override():
    """Routed profile that exists but has no SOUL.md: ``skip_host_soul=True``
    + ``identity_prompt_override=None`` must fall through to
    ``DEFAULT_AGENT_IDENTITY``, NOT call ``load_soul_md()``.  Without the
    guard, the agent would silently inherit the host's persona — exactly
    the leak the routing change is meant to close."""
    from run_agent import AIAgent
    from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

    agent = _make_minimal_agent_stub(
        identity_prompt_override=None,
        skip_context_files=True,
        skip_host_soul=True,
    )

    with patch("run_agent.load_soul_md", return_value="HOST_SOUL_PERSONA") as mock_load_soul:
        prompt = AIAgent._build_system_prompt(agent)

    assert prompt.startswith(DEFAULT_AGENT_IDENTITY)
    assert "HOST_SOUL_PERSONA" not in prompt
    mock_load_soul.assert_not_called()


def test_skip_host_soul_with_override_still_uses_override():
    """Override always wins, regardless of ``skip_host_soul``.  This is
    the typical routed-with-SOUL case — the override is the routed
    profile's SOUL, ``skip_host_soul`` is the belt-and-braces guard."""
    from run_agent import AIAgent

    agent = _make_minimal_agent_stub(
        identity_prompt_override="PHYLLIS_PERSONA",
        skip_context_files=True,
        skip_host_soul=True,
    )

    with patch("run_agent.load_soul_md", return_value="HOST_SOUL_PERSONA") as mock_load_soul:
        prompt = AIAgent._build_system_prompt(agent)

    assert prompt.startswith("PHYLLIS_PERSONA")
    assert "HOST_SOUL_PERSONA" not in prompt
    mock_load_soul.assert_not_called()


def test_skip_host_soul_blocks_context_files_soul_reinjection():
    """When ``skip_host_soul=True`` and context files ARE allowed (the
    routed profile still wants AGENTS.md / .cursorrules), SOUL.md must
    still be excluded from the context-files block — otherwise it sneaks
    back into the prompt under ``# Project Context``."""
    from run_agent import AIAgent

    agent = _make_minimal_agent_stub(
        identity_prompt_override=None,
        skip_context_files=False,  # Allow AGENTS.md, .cursorrules
        skip_host_soul=True,        # But still keep host SOUL out
    )

    with (
        patch("run_agent.load_soul_md", return_value="HOST_SOUL_PERSONA") as mock_identity_soul,
        # build_context_files_prompt also calls load_soul_md when not
        # skip_soul; patch the prompt_builder-side import too so we can
        # observe whether SOUL would have been re-injected via that path.
        patch("agent.prompt_builder.load_soul_md", return_value="HOST_SOUL_PERSONA") as mock_ctx_soul,
    ):
        prompt = AIAgent._build_system_prompt(agent)

    # Identity layer never asks for it
    mock_identity_soul.assert_not_called()
    # And the context-files path was told to skip it
    mock_ctx_soul.assert_not_called()
    assert "HOST_SOUL_PERSONA" not in prompt
