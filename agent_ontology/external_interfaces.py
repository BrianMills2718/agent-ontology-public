"""Typed adapter layer for external interface calls.

This module provides a single dispatch surface for generated agents when they
interact with external systems (search APIs, MCP tools, RPC-like side effects).

Modes:
- deterministic (default): no network calls; stable synthetic outputs
- live: bypass adapter and let generated tool logic call real systems
- llm_mock: scaffold mode for LLM-backed external simulation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
import re
import shlex
import subprocess
from typing import Any, Literal, Protocol


MockMode = Literal["deterministic", "live", "llm_mock"]
_VALID_MOCK_MODES: set[str] = {"deterministic", "live", "llm_mock"}


@dataclass(slots=True)
class ExternalCall:
    """Typed external interface request."""

    interface_name: str
    payload: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None

    def __post_init__(self) -> None:
        self.interface_name = str(self.interface_name or "").strip()
        if not isinstance(self.payload, dict):
            self.payload = {"value": self.payload}
        if not isinstance(self.context, dict):
            self.context = {"value": self.context}
        if self.trace_id is not None:
            self.trace_id = str(self.trace_id)


@dataclass(slots=True)
class ExternalResult:
    """Typed external interface response."""

    handled: bool
    ok: bool
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    provider: str = "none"
    mode: MockMode = "deterministic"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["text"] = self.text or ""
        return payload


class ExternalAdapter(Protocol):
    """Adapter contract for external interface simulation."""

    def handle(self, call: ExternalCall) -> ExternalResult:
        """Handle an external call and return a typed result."""


def _normalize_mode(mode: str | None) -> MockMode:
    raw = str(mode or os.environ.get("AGENT_ONTOLOGY_MOCK_MODE", "deterministic")).strip().lower()
    if raw in {"off", "none", "disabled"}:
        raw = "live"
    if raw not in _VALID_MOCK_MODES:
        raw = "deterministic"
    return raw  # type: ignore[return-value]


def get_mock_mode() -> MockMode:
    """Return current external interface mode."""

    return _normalize_mode(None)


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9_]+", text.lower()) if len(tok) >= 3}


_DETERMINISTIC_CORPUS: list[dict[str, str]] = [
    {
        "title": "LiveKit Runtime Model",
        "snippet": "Session lifecycle tracks room_connected, session_created, and handler registration before response synthesis.",
    },
    {
        "title": "RPC Event Handling",
        "snippet": "RPC methods like interrupt_speech and toggle_streaming update runtime state and publish structured acknowledgements.",
    },
    {
        "title": "Semantic Search Delegation",
        "snippet": "MCP task planning can delegate external retrieval through semantic_search with normalized query payloads.",
    },
    {
        "title": "State Diff Telemetry",
        "snippet": "Per-step state/store diffs capture added, changed, and removed keys for runtime observability.",
    },
    {
        "title": "Deterministic Mocking",
        "snippet": "Deterministic mocks avoid network variance and provide repeatable fixtures for external-system behavior.",
    },
    {
        "title": "Final Output Projection",
        "snippet": "Final output inference can select from channel messages, state.data keys, and synthesized fallback previews.",
    },
]


def _rank_docs(query: str, limit: int = 5) -> list[dict[str, str]]:
    q = str(query or "")
    q_tokens = _tokenize(q)
    scored: list[tuple[int, str, dict[str, str]]] = []
    for doc in _DETERMINISTIC_CORPUS:
        title = doc.get("title", "")
        snippet = doc.get("snippet", "")
        doc_tokens = _tokenize(f"{title} {snippet}")
        overlap = len(q_tokens & doc_tokens)
        # Keep tie-break deterministic and stable.
        tie = hashlib.sha1(f"{q}|{title}".encode("utf-8")).hexdigest()
        scored.append((overlap, tie, doc))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [doc for _, _, doc in scored][: max(1, int(limit))]
    return ranked


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_mcp_command(
    *,
    server: str,
    explicit_cmd: str | None = None,
    allow_default_server_pkg: bool = False,
) -> tuple[list[str], str]:
    if explicit_cmd:
        cmd = shlex.split(str(explicit_cmd))
        if cmd:
            return cmd, "explicit_cmd"

    server_norm = str(server or "").strip()
    env_key = f"MCP_SERVER_{server_norm.upper()}" if server_norm else ""
    if env_key:
        env_cmd = str(os.environ.get(env_key, "")).strip()
        if env_cmd:
            cmd = shlex.split(env_cmd)
            if cmd:
                return cmd, env_key

    if server_norm and allow_default_server_pkg:
        return ["npx", "-y", f"@modelcontextprotocol/server-{server_norm}"], f"default_pkg:{server_norm}"

    raise ValueError(
        "MCP command not configured. Set AGENT_ONTOLOGY_SEMANTIC_SEARCH_MCP_CMD or "
        f"{env_key or 'MCP_SERVER_<NAME>'}."
    )


def _parse_json_from_stdout(stdout_text: str) -> dict[str, Any] | list[Any] | None:
    text = str(stdout_text or "").strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass

    # Some servers log extra lines; try parsing line-wise from bottom-up.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        if not line or line[0] not in "[{":
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return parsed

    # Last attempt: parse largest {...} span.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        maybe = text[start : end + 1]
        try:
            parsed = json.loads(maybe)
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def _extract_mcp_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("text", "summary", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        content = payload.get("content")
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunk = item.strip()
                    if chunk:
                        chunks.append(chunk)
                    continue
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
                    continue
                if item.get("type") == "text":
                    text = item.get("content")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
            if chunks:
                return "\n".join(chunks)
    if isinstance(payload, list):
        parts = [_extract_mcp_text(item) for item in payload]
        joined = "\n".join([part for part in parts if part])
        if joined:
            return joined
    return ""


class DeterministicExternalAdapter:
    """Deterministic adapter: stable outputs, no network."""

    mode: MockMode = "deterministic"

    def handle(self, call: ExternalCall) -> ExternalResult:
        name = call.interface_name
        payload = call.payload

        if name in {"semantic_search", "search", "web_fetch"}:
            query = str(payload.get("query", payload.get("input", "")))
            top = _rank_docs(query, limit=5)
            lines = [f"{row['title']}: {row['snippet']}" for row in top]
            return ExternalResult(
                handled=True,
                ok=True,
                text="\n".join(lines),
                data={"query": query, "results": top},
                provider="deterministic_corpus",
                mode=self.mode,
            )

        if name in {"lookup", "wikipedia_lookup"}:
            query = str(payload.get("query", payload.get("input", "")))
            top = _rank_docs(query, limit=1)
            row = top[0]
            text = f"{row['title']}: {row['snippet']}"
            return ExternalResult(
                handled=True,
                ok=True,
                text=text,
                data={"query": query, "result": row},
                provider="deterministic_corpus",
                mode=self.mode,
            )

        if name in {"mcp_tool_call", "mcp_call"}:
            tool_name = str(payload.get("tool") or payload.get("tool_name") or "unknown_tool")
            args = payload.get("args", {})
            result_payload = {
                "status": "ok",
                "tool": tool_name,
                "args": args if isinstance(args, dict) else {"value": args},
                "mocked": True,
            }
            return ExternalResult(
                handled=True,
                ok=True,
                text=json.dumps(result_payload, ensure_ascii=False),
                data=result_payload,
                provider="deterministic_mcp",
                mode=self.mode,
            )

        if name in {"rpc_event", "rpc_emit"}:
            method = str(payload.get("rpc_method") or payload.get("event_name") or "rpc_event")
            return ExternalResult(
                handled=True,
                ok=True,
                text=f"RPC acknowledged: {method}",
                data={"acknowledged": True, "method": method, "mocked": True},
                provider="deterministic_rpc",
                mode=self.mode,
            )

        return ExternalResult(handled=False, ok=True, provider="deterministic_passthrough", mode=self.mode)


class LiveExternalAdapter:
    """Live adapter with optional MCP-backed handling."""

    mode: MockMode = "live"

    def _call_mcp_tool(
        self,
        *,
        server: str,
        tool_name: str,
        tool_args: dict[str, Any],
        explicit_cmd: str | None = None,
        trace_id: str | None = None,
        allow_default_server_pkg: bool = False,
    ) -> ExternalResult:
        try:
            cmd, cmd_source = _resolve_mcp_command(
                server=server,
                explicit_cmd=explicit_cmd,
                allow_default_server_pkg=allow_default_server_pkg,
            )
        except ValueError as exc:
            return ExternalResult(
                handled=True,
                ok=False,
                error=str(exc),
                provider=f"live_mcp:{server or 'unset'}",
                mode=self.mode,
            )

        timeout_seconds = int(
            str(os.environ.get("AGENT_ONTOLOGY_MCP_TIMEOUT_SECONDS", "30") or "30")
        )
        request = {
            "jsonrpc": "2.0",
            "id": trace_id or "agent_ontology_live",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": tool_args,
            },
        }

        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=max(1, timeout_seconds),
            )
        except FileNotFoundError:
            return ExternalResult(
                handled=True,
                ok=False,
                error=f"MCP executable not found: {cmd[0]}",
                provider=f"live_mcp:{server or 'unset'}",
                mode=self.mode,
            )
        except subprocess.TimeoutExpired:
            return ExternalResult(
                handled=True,
                ok=False,
                error=f"MCP call timed out after {timeout_seconds}s",
                provider=f"live_mcp:{server or 'unset'}",
                mode=self.mode,
            )
        except Exception as exc:
            return ExternalResult(
                handled=True,
                ok=False,
                error=f"MCP execution failed: {type(exc).__name__}: {exc}",
                provider=f"live_mcp:{server or 'unset'}",
                mode=self.mode,
            )

        if proc.returncode != 0:
            stderr = str(proc.stderr or "").strip()
            stdout = str(proc.stdout or "").strip()
            err = stderr or stdout or f"exit code {proc.returncode}"
            return ExternalResult(
                handled=True,
                ok=False,
                error=f"MCP call failed ({proc.returncode}): {err}",
                provider=f"live_mcp:{server or 'unset'}",
                mode=self.mode,
                data={
                    "cmd_source": cmd_source,
                    "server": server,
                    "tool": tool_name,
                    "args": tool_args,
                },
            )

        parsed = _parse_json_from_stdout(str(proc.stdout or ""))
        if parsed is None:
            return ExternalResult(
                handled=True,
                ok=False,
                error="MCP call returned non-JSON output",
                provider=f"live_mcp:{server or 'unset'}",
                mode=self.mode,
                data={
                    "cmd_source": cmd_source,
                    "server": server,
                    "tool": tool_name,
                    "args": tool_args,
                    "stdout": str(proc.stdout or ""),
                    "stderr": str(proc.stderr or ""),
                },
            )

        # JSON-RPC response payload.
        result_payload: Any = parsed
        if isinstance(parsed, dict):
            if parsed.get("error") is not None:
                return ExternalResult(
                    handled=True,
                    ok=False,
                    error=f"MCP JSON-RPC error: {parsed.get('error')}",
                    provider=f"live_mcp:{server or 'unset'}",
                    mode=self.mode,
                    data={
                        "cmd_source": cmd_source,
                        "server": server,
                        "tool": tool_name,
                        "args": tool_args,
                        "response": parsed,
                    },
                )
            if "result" in parsed:
                result_payload = parsed.get("result")

        text = _extract_mcp_text(result_payload)
        if not text:
            try:
                text = json.dumps(result_payload, ensure_ascii=False)
            except Exception:
                text = str(result_payload)

        return ExternalResult(
            handled=True,
            ok=True,
            text=text,
            data={
                "cmd_source": cmd_source,
                "server": server,
                "tool": tool_name,
                "args": tool_args,
                "result": result_payload,
            },
            provider=f"live_mcp:{server or 'unset'}",
            mode=self.mode,
        )

    def _maybe_handle_semantic_search(self, call: ExternalCall) -> ExternalResult:
        payload = call.payload
        query = str(payload.get("query", payload.get("input", payload.get("text", ""))) or "").strip()
        if not query:
            return ExternalResult(
                handled=True,
                ok=False,
                error="Missing query for semantic_search",
                provider="live_mcp:semantic_search",
                mode=self.mode,
            )

        explicit_cmd = str(os.environ.get("AGENT_ONTOLOGY_SEMANTIC_SEARCH_MCP_CMD", "")).strip() or None
        explicit_server = str(payload.get("server") or os.environ.get("AGENT_ONTOLOGY_SEMANTIC_SEARCH_SERVER", "")).strip()
        explicit_tool = str(payload.get("tool") or payload.get("tool_name") or os.environ.get("AGENT_ONTOLOGY_SEMANTIC_SEARCH_TOOL", "")).strip()
        configured = bool(explicit_cmd or explicit_server or explicit_tool)
        allow_default = _truthy_env("AGENT_ONTOLOGY_ENABLE_DEFAULT_MCP_SEARCH", default=False)

        # Preserve passthrough behavior unless MCP search is explicitly configured.
        if not configured and not allow_default:
            return ExternalResult(handled=False, ok=True, provider="live_passthrough", mode=self.mode)

        server = explicit_server or "search"
        tool_name = explicit_tool or "search"
        args = payload.get("args")
        tool_args = dict(args) if isinstance(args, dict) else {}
        tool_args.setdefault("query", query)
        if payload.get("limit") is not None and "limit" not in tool_args:
            tool_args["limit"] = payload.get("limit")

        return self._call_mcp_tool(
            server=server,
            tool_name=tool_name,
            tool_args=tool_args,
            explicit_cmd=explicit_cmd,
            trace_id=call.trace_id,
            allow_default_server_pkg=allow_default,
        )

    def _maybe_handle_mcp_call(self, call: ExternalCall) -> ExternalResult:
        payload = call.payload
        server = str(payload.get("server", "") or "").strip()
        tool_name = str(payload.get("tool") or payload.get("tool_name") or "").strip()
        args = payload.get("args")
        tool_args = dict(args) if isinstance(args, dict) else {}
        explicit_cmd = str(payload.get("command") or payload.get("cmd") or "").strip() or None

        if not (tool_name and (explicit_cmd or server)):
            return ExternalResult(handled=False, ok=True, provider="live_passthrough", mode=self.mode)

        return self._call_mcp_tool(
            server=server,
            tool_name=tool_name,
            tool_args=tool_args,
            explicit_cmd=explicit_cmd,
            trace_id=call.trace_id,
            allow_default_server_pkg=False,
        )

    def handle(self, call: ExternalCall) -> ExternalResult:
        name = call.interface_name
        if name in {"semantic_search", "search", "web_fetch", "lookup", "wikipedia_lookup"}:
            return self._maybe_handle_semantic_search(call)
        if name in {"mcp_tool_call", "mcp_call"}:
            return self._maybe_handle_mcp_call(call)
        return ExternalResult(handled=False, ok=True, provider="live_passthrough", mode=self.mode)


class LLMMockExternalAdapter:
    """LLM-backed mock scaffold for external calls.

    This mode is intentionally lightweight and best-effort. It allows the same
    typed adapter contract while deferring richer mocking behavior.
    """

    mode: MockMode = "llm_mock"

    def __init__(self, model: str | None = None):
        self.model = model or os.environ.get("AGENT_ONTOLOGY_LLM_MOCK_MODEL", "openrouter/deepseek/deepseek-chat")

    def handle(self, call: ExternalCall) -> ExternalResult:
        supported = {"semantic_search", "search", "lookup", "wikipedia_lookup", "mcp_tool_call", "mcp_call", "rpc_event", "rpc_emit"}
        if call.interface_name not in supported:
            return ExternalResult(handled=False, ok=True, provider="llm_mock_passthrough", mode=self.mode)

        try:
            try:
                from llm_client import call_llm as llm_call
            except ImportError:
                from llm_client import call_llm as llm_call  # type: ignore[no-redef]
        except Exception as exc:
            return ExternalResult(
                handled=True,
                ok=False,
                error=f"LLM mock unavailable: {type(exc).__name__}: {exc}",
                provider=f"llm_mock:{self.model}",
                mode=self.mode,
            )

        system = (
            "You simulate an external system for testing. "
            "Return concise plain text describing the synthetic result."
        )
        user = json.dumps(
            {
                "interface_name": call.interface_name,
                "payload": call.payload,
                "context": call.context,
                "trace_id": call.trace_id,
            },
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            result = llm_call(
                self.model,
                messages,
                task="agent_ontology.external_llm_mock",
                trace_id=call.trace_id or "agent_ontology/external_llm_mock",
                max_budget=float(os.environ.get("AGENT_ONTOLOGY_MAX_BUDGET_USD", "0") or 0),
                max_tokens=400,
                num_retries=1,
            )
            text = (result.content or "").strip()
            return ExternalResult(
                handled=True,
                ok=True,
                text=text,
                data={"model": self.model, "interface_name": call.interface_name},
                provider=f"llm_mock:{self.model}",
                mode=self.mode,
            )
        except Exception as exc:
            return ExternalResult(
                handled=True,
                ok=False,
                error=f"LLM mock call failed: {type(exc).__name__}: {exc}",
                provider=f"llm_mock:{self.model}",
                mode=self.mode,
            )


_ADAPTER_CACHE: dict[str, ExternalAdapter] = {}


def get_external_adapter(mode: str | None = None) -> ExternalAdapter:
    """Return adapter instance for mode."""

    norm = _normalize_mode(mode)
    adapter = _ADAPTER_CACHE.get(norm)
    if adapter is not None:
        return adapter

    if norm == "live":
        adapter = LiveExternalAdapter()
    elif norm == "llm_mock":
        adapter = LLMMockExternalAdapter()
    else:
        adapter = DeterministicExternalAdapter()

    _ADAPTER_CACHE[norm] = adapter
    return adapter


def dispatch_external_interface(
    interface_name: str,
    payload: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    *,
    mode: str | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Dispatch a typed external call and return a plain dict result."""

    call = ExternalCall(
        interface_name=interface_name,
        payload=payload or {},
        context=context or {},
        trace_id=trace_id,
    )
    adapter = get_external_adapter(mode)
    result = adapter.handle(call)
    out = result.to_dict()
    out["interface_name"] = call.interface_name
    return out
