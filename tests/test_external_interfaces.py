from __future__ import annotations

from types import SimpleNamespace

import agent_ontology.external_interfaces as ext
from agent_ontology.external_interfaces import (
    DeterministicExternalAdapter,
    ExternalCall,
    LiveExternalAdapter,
    dispatch_external_interface,
    get_mock_mode,
)


def test_deterministic_semantic_search_returns_stable_text() -> None:
    adapter = DeterministicExternalAdapter()
    call_result = adapter.handle(ExternalCall(interface_name="semantic_search", payload={"query": "livekit rpc"}))
    assert call_result.handled is True
    assert call_result.ok is True
    assert "LiveKit" in call_result.text or "RPC" in call_result.text


def test_deterministic_web_fetch_aliases_to_semantic_search() -> None:
    adapter = DeterministicExternalAdapter()
    call_result = adapter.handle(ExternalCall(interface_name="web_fetch", payload={"query": "mcp delegation"}))
    assert call_result.handled is True
    assert call_result.ok is True
    assert call_result.provider == "deterministic_corpus"
    assert "query" in call_result.data


def test_live_adapter_passthrough() -> None:
    adapter = LiveExternalAdapter()
    call_result = adapter.handle(ExternalCall(interface_name="semantic_search", payload={"query": "anything"}))
    assert call_result.handled is False
    assert call_result.ok is True


def test_dispatch_mode_defaults_to_deterministic(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_ONTOLOGY_MOCK_MODE", raising=False)
    assert get_mock_mode() == "deterministic"
    out = dispatch_external_interface("mcp_tool_call", {"tool": "calendar.create"})
    assert out["handled"] is True
    assert out["ok"] is True
    assert out["provider"] == "deterministic_mcp"


def test_dispatch_live_mode_bypasses(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ONTOLOGY_MOCK_MODE", "live")
    out = dispatch_external_interface("semantic_search", {"query": "test"})
    assert out["handled"] is False
    assert out["ok"] is True


def test_live_semantic_search_uses_configured_mcp_cmd(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ONTOLOGY_MOCK_MODE", "live")
    monkeypatch.setenv("AGENT_ONTOLOGY_SEMANTIC_SEARCH_MCP_CMD", "mcp-search --stdio")

    calls: list[dict[str, object]] = []

    def _fake_run(cmd, input, capture_output, text, timeout):  # noqa: ANN001
        calls.append(
            {
                "cmd": cmd,
                "input": input,
                "capture_output": capture_output,
                "text": text,
                "timeout": timeout,
            }
        )
        return SimpleNamespace(
            returncode=0,
            stdout='{"jsonrpc":"2.0","id":"x","result":{"content":[{"type":"text","text":"mock semantic hit"}]}}',
            stderr="",
        )

    monkeypatch.setattr(ext.subprocess, "run", _fake_run)

    out = dispatch_external_interface("semantic_search", {"query": "livekit"}, mode="live", trace_id="t1")
    assert out["handled"] is True
    assert out["ok"] is True
    assert "mock semantic hit" in out["text"]
    assert out["provider"].startswith("live_mcp:")
    assert len(calls) == 1
    req = calls[0]["input"]
    assert isinstance(req, str)
    assert '"method": "tools/call"' in req
    assert '"query": "livekit"' in req


def test_live_mcp_call_reports_non_json_output(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_ONTOLOGY_MOCK_MODE", "live")
    monkeypatch.setenv("MCP_SERVER_DEMO", "mcp-demo --stdio")

    def _fake_run(cmd, input, capture_output, text, timeout):  # noqa: ANN001
        _ = (cmd, input, capture_output, text, timeout)
        return SimpleNamespace(returncode=0, stdout="not json", stderr="")

    monkeypatch.setattr(ext.subprocess, "run", _fake_run)

    out = dispatch_external_interface(
        "mcp_call",
        {"server": "demo", "tool": "search", "args": {"query": "test"}},
        mode="live",
    )
    assert out["handled"] is True
    assert out["ok"] is False
    assert "non-JSON" in (out.get("error") or "")
