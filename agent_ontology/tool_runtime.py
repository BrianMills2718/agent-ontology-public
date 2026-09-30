"""Shared local tool runtime for generated agents.

This module provides a best-effort, typed-ish adapter layer for tool calls that
are declared in specs but not hardcoded in generated tool implementations.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import select
import shutil
import subprocess
import sys
import threading
import time
import uuid
import hashlib
from collections import defaultdict
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


ToolResult = dict[str, Any]

_SESSION_LOCK = threading.Lock()
_SESSIONS: dict[int, subprocess.Popen[str]] = {}
_NEXT_SESSION_ID = 1
_RUNTIME_STATE: dict[str, Any] = defaultdict(dict)
_TRACE_WRITTEN_RE = re.compile(
    r"^Trace written to (?P<path>.+?) \((?P<calls>\d+) calls, (?P<dur>\d+)ms total\)"
)

def _infer_child_run_id_from_trace_path(path: pathlib.Path) -> str | None:
    """Infer child live-run id from a trace path under .live_runs/<run_id>/trace.json."""
    try:
        parts = list(path.resolve().parts)
    except Exception:
        parts = list(path.parts)
    if ".live_runs" not in parts:
        return None
    idx = parts.index(".live_runs")
    if idx + 1 >= len(parts):
        return None
    candidate = str(parts[idx + 1]).strip()
    if not candidate:
        return None
    return candidate


def _ok(output: str, *, data: dict[str, Any] | None = None) -> ToolResult:
    return {
        "handled": True,
        "ok": True,
        "output": output,
        "data": data or {},
    }


def _err(message: str, *, data: dict[str, Any] | None = None) -> ToolResult:
    return {
        "handled": True,
        "ok": False,
        "output": message,
        "error": message,
        "data": data or {},
    }


def _normalize_tool_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    if normalized.startswith("functions."):
        normalized = normalized.split(".", 1)[1]
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")
    return normalized


def _parse_input(input_text: str) -> tuple[dict[str, Any], Any]:
    raw = str(input_text or "")
    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed, parsed
            return {"value": parsed}, parsed
        except json.JSONDecodeError:
            pass
    return {"value": raw}, raw


def _workdir_from_payload(payload: dict[str, Any]) -> str:
    value = payload.get("workdir") or payload.get("cwd") or os.environ.get("AGENT_ONTOLOGY_TOOL_WORKDIR")
    if not value:
        return os.getcwd()
    return str(value)


def _read_nonblocking(proc: subprocess.Popen[str], *, timeout_ms: int = 100, max_chars: int = 12000) -> str:
    if proc.stdout is None:
        return ""
    fd = proc.stdout.fileno()
    chunks: list[str] = []
    deadline = time.time() + max(0, timeout_ms) / 1000.0
    while time.time() < deadline and sum(len(c) for c in chunks) < max_chars:
        remaining = max(0.0, deadline - time.time())
        readable, _, _ = select.select([fd], [], [], remaining)
        if not readable:
            break
        data = os.read(fd, 4096)
        if not data:
            break
        chunks.append(data.decode("utf-8", errors="replace"))
    return "".join(chunks)


def _coerce_cmd(payload: dict[str, Any], fallback: Any) -> str:
    cmd = payload.get("cmd")
    if not cmd and isinstance(fallback, str):
        cmd = fallback
    if not cmd:
        return ""
    return str(cmd).strip()


def _tool_exec_command(payload: dict[str, Any], fallback: Any) -> ToolResult:
    global _NEXT_SESSION_ID
    cmd = _coerce_cmd(payload, fallback)
    if not cmd:
        return _err("Missing cmd for exec_command")

    timeout_ms = int(payload.get("timeout_ms") or payload.get("yield_time_ms") or 30000)
    workdir = _workdir_from_payload(payload)
    tty = bool(payload.get("tty", False))

    if tty:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            cwd=workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        with _SESSION_LOCK:
            session_id = _NEXT_SESSION_ID
            _NEXT_SESSION_ID += 1
            _SESSIONS[session_id] = proc
        output = _read_nonblocking(proc, timeout_ms=min(timeout_ms, 1000))
        running = proc.poll() is None
        return _ok(
            json.dumps(
                {
                    "session_id": session_id,
                    "running": running,
                    "output": output,
                },
                ensure_ascii=False,
            ),
            data={"session_id": session_id, "running": running, "output": output},
        )

    timeout_seconds = max(1, int(timeout_ms / 1000))
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return _err(f"Command timed out after {timeout_seconds}s")
    except Exception as exc:
        return _err(f"exec_command error: {type(exc).__name__}: {exc}")

    response = {
        "exit_code": result.returncode,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }
    text = json.dumps(response, ensure_ascii=False)
    if result.returncode != 0:
        return _err(text, data=response)
    return _ok(text, data=response)


def _tool_write_stdin(payload: dict[str, Any]) -> ToolResult:
    session_id = payload.get("session_id")
    if session_id is None:
        return _err("write_stdin requires session_id")
    try:
        session_id_int = int(session_id)
    except Exception:
        return _err(f"Invalid session_id: {session_id}")

    with _SESSION_LOCK:
        proc = _SESSIONS.get(session_id_int)
    if proc is None:
        return _err(f"Unknown session_id: {session_id_int}")

    chars = str(payload.get("chars", "") or "")
    timeout_ms = int(payload.get("yield_time_ms") or 200)
    if proc.stdin is None:
        return _err(f"Session {session_id_int} has no writable stdin")
    try:
        if chars:
            proc.stdin.write(chars.encode("utf-8"))
            proc.stdin.flush()
    except Exception as exc:
        return _err(f"stdin write failed: {type(exc).__name__}: {exc}")

    output = _read_nonblocking(proc, timeout_ms=timeout_ms)
    running = proc.poll() is None
    if not running:
        with _SESSION_LOCK:
            _SESSIONS.pop(session_id_int, None)
    data = {
        "session_id": session_id_int,
        "running": running,
        "exit_code": proc.poll(),
        "output": output,
    }
    return _ok(json.dumps(data, ensure_ascii=False), data=data)


def _tool_read(payload: dict[str, Any], fallback: Any) -> ToolResult:
    path = payload.get("path") or payload.get("file_path") or payload.get("file") or payload.get("filename") or fallback
    if not path:
        return _err("read requires path")
    target = pathlib.Path(str(path))
    try:
        content = target.read_text(encoding="utf-8")
        max_chars = int(payload.get("max_chars") or 20000)
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...[truncated]"
        return _ok(content, data={"path": str(target)})
    except Exception as exc:
        return _err(f"Read error for {target}: {type(exc).__name__}: {exc}")


def _tool_write(payload: dict[str, Any], fallback: Any) -> ToolResult:
    path = payload.get("path") or payload.get("file_path") or payload.get("file") or payload.get("filename")
    content = payload.get("content", payload.get("contents", payload.get("text")))
    append = bool(payload.get("append", False))
    mode = str(payload.get("mode", "")).strip().lower()
    if mode == "append":
        append = True
    if path is None and isinstance(fallback, dict):
        path = (
            fallback.get("path")
            or fallback.get("file_path")
            or fallback.get("file")
            or fallback.get("filename")
        )
        content = fallback.get("content", fallback.get("contents", fallback.get("text", content)))
        if "append" in fallback:
            append = bool(fallback.get("append"))
    if path is None:
        return _err("write requires path")
    if content is None:
        content = ""
    target = pathlib.Path(str(path))
    content_text = str(content)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if append:
            with target.open("a", encoding="utf-8") as fh:
                fh.write(content_text)
            action = "Appended"
        else:
            target.write_text(content_text, encoding="utf-8")
            action = "Wrote"
        return _ok(f"{action} {target}", data={"path": str(target), "bytes": len(content_text), "append": append})
    except Exception as exc:
        return _err(f"Write error for {target}: {type(exc).__name__}: {exc}")


def _tool_edit(payload: dict[str, Any], fallback: Any) -> ToolResult:
    if isinstance(fallback, dict):
        merged = dict(fallback)
        merged.update(payload)
        payload = merged
    path = payload.get("path") or payload.get("file_path") or payload.get("file") or payload.get("filename")
    old_text = payload.get("oldText", payload.get("old", payload.get("find")))
    new_text = payload.get("newText", payload.get("new", payload.get("replace")))
    replace_all = bool(payload.get("all", False))
    if not path:
        return _err("edit requires path")
    if old_text is None or new_text is None:
        return _err("edit requires old/new text")
    target = pathlib.Path(str(path))
    try:
        original = target.read_text(encoding="utf-8")
    except Exception as exc:
        return _err(f"Edit read error for {target}: {type(exc).__name__}: {exc}")
    if str(old_text) not in original:
        return _err(f"Edit target text not found in {target}")
    if replace_all:
        updated = original.replace(str(old_text), str(new_text))
    else:
        updated = original.replace(str(old_text), str(new_text), 1)
    try:
        target.write_text(updated, encoding="utf-8")
    except Exception as exc:
        return _err(f"Edit write error for {target}: {type(exc).__name__}: {exc}")
    return _ok(f"Edited {target}", data={"path": str(target)})


def _tool_glob(payload: dict[str, Any], fallback: Any) -> ToolResult:
    pattern = payload.get("pattern") or payload.get("glob") or (fallback if isinstance(fallback, str) else "")
    if not pattern:
        return _err("glob requires pattern")
    root = pathlib.Path(str(payload.get("path") or payload.get("root") or "."))
    if not root.exists():
        return _err(f"glob root not found: {root}")
    try:
        matches = sorted(str(p) for p in root.glob(str(pattern)))
    except Exception as exc:
        return _err(f"glob error: {type(exc).__name__}: {exc}")
    limit = int(payload.get("limit") or 200)
    matches = matches[: max(1, limit)]
    return _ok("\n".join(matches), data={"count": len(matches), "matches": matches})


def _tool_grep(payload: dict[str, Any], fallback: Any) -> ToolResult:
    pattern = payload.get("pattern") or (fallback if isinstance(fallback, str) else "")
    path = str(payload.get("path") or ".")
    if not pattern:
        return _err("grep requires pattern")
    rg_bin = shutil.which("rg")
    if rg_bin:
        try:
            result = subprocess.run(
                [rg_bin, "-n", pattern, path],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if result.returncode in (0, 1):
                output = (result.stdout or "").strip()
                return _ok(output, data={"path": path, "pattern": pattern})
            return _err((result.stderr or result.stdout or "rg error").strip())
        except Exception as exc:
            return _err(f"grep error: {type(exc).__name__}: {exc}")

    # Fallback without ripgrep.
    root = pathlib.Path(path)
    if not root.exists():
        return _err(f"grep path not found: {path}")
    rx = re.compile(pattern)
    lines: list[str] = []
    try:
        for fp in root.rglob("*"):
            if not fp.is_file():
                continue
            try:
                content = fp.read_text(encoding="utf-8")
            except Exception:
                continue
            for idx, line in enumerate(content.splitlines(), 1):
                if rx.search(line):
                    lines.append(f"{fp}:{idx}:{line}")
                    if len(lines) >= 200:
                        break
            if len(lines) >= 200:
                break
    except re.error as exc:
        return _err(f"Invalid regex: {exc}")
    return _ok("\n".join(lines), data={"matches": len(lines)})


def _tool_web_fetch(payload: dict[str, Any], fallback: Any) -> ToolResult:
    url = payload.get("url") or (fallback if isinstance(fallback, str) else "")
    if not url:
        return _err("web_fetch requires url")
    url_text = str(url).strip()
    if not re.match(r"^https?://", url_text, flags=re.IGNORECASE):
        # Treat non-URL text as a search query through external adapter when available.
        try:
            from agent_ontology.external_interfaces import dispatch_external_interface
        except Exception:
            dispatch_external_interface = None  # type: ignore[assignment]
        if dispatch_external_interface is not None:
            try:
                ext = dispatch_external_interface(
                    "semantic_search",
                    payload={"query": url_text},
                    context={"source": "tool_runtime.web_fetch"},
                )
                if isinstance(ext, dict) and ext.get("handled"):
                    if not ext.get("ok", True):
                        return _err(str(ext.get("error", "external search failure")))
                    text = str(ext.get("text", "") or "")
                    if text:
                        return _ok(text, data={"query": url_text, "provider": ext.get("provider")})
                    data = ext.get("data", {})
                    return _ok(json.dumps(data, ensure_ascii=False), data={"query": url_text, "data": data})
            except Exception as exc:
                return _err(f"search adapter error: {type(exc).__name__}: {exc}")
        return _err(f"web_fetch expected URL, got query text: {url_text}")
    req = Request(url_text, headers={"User-Agent": "AgentOntology/1.0"})
    try:
        with urlopen(req, timeout=15) as resp:  # nosec - explicit test/runtime fetch
            body = resp.read().decode("utf-8", errors="replace")
            max_chars = int(payload.get("max_chars") or 12000)
            if len(body) > max_chars:
                body = body[:max_chars] + "\n...[truncated]"
            return _ok(body, data={"url": url_text, "status": getattr(resp, "status", None)})
    except URLError as exc:
        return _err(f"web_fetch error for {url_text}: {exc}")
    except Exception as exc:
        return _err(f"web_fetch error for {url_text}: {type(exc).__name__}: {exc}")


def _dispatch_external_tool(
    interface_name: str,
    payload: dict[str, Any] | None = None,
    *,
    context: dict[str, Any] | None = None,
) -> ToolResult:
    try:
        from agent_ontology.external_interfaces import dispatch_external_interface
    except Exception as exc:
        return _err(f"external adapter unavailable: {type(exc).__name__}: {exc}")

    mode = os.environ.get("AGENT_ONTOLOGY_MOCK_MODE", "deterministic")
    trace_id = str((payload or {}).get("trace_id") or "").strip() or None
    try:
        out = dispatch_external_interface(
            interface_name,
            payload=payload or {},
            context=context or {"source": "tool_runtime"},
            mode=mode,
            trace_id=trace_id,
        )
    except Exception as exc:
        return _err(f"external adapter error: {type(exc).__name__}: {exc}")

    if not isinstance(out, dict):
        return _err(f"external adapter returned non-dict for {interface_name}")
    handled = bool(out.get("handled", False))
    ok = bool(out.get("ok", True))
    text = str(out.get("text", "") or "")
    if not handled:
        return _err(f"external adapter did not handle interface: {interface_name}", data=out)
    if not ok:
        return _err(str(out.get("error", f"external interface failed: {interface_name}")), data=out)
    if text.strip():
        return _ok(text, data=out)
    return _ok(json.dumps(out, ensure_ascii=False), data=out)


def _coerce_query_payload(payload: dict[str, Any], fallback: Any) -> dict[str, Any]:
    query = (
        payload.get("query")
        or payload.get("input")
        or payload.get("text")
        or payload.get("prompt")
        or payload.get("task")
    )
    if query is None and isinstance(fallback, str):
        query = fallback
    if query is None and isinstance(fallback, dict):
        query = (
            fallback.get("query")
            or fallback.get("input")
            or fallback.get("text")
            or fallback.get("prompt")
            or fallback.get("task")
        )
    out = dict(payload)
    out["query"] = str(query or "").strip()
    if out["query"] == "":
        out.pop("query", None)
    return out


def _tool_search(payload: dict[str, Any], fallback: Any) -> ToolResult:
    req = _coerce_query_payload(payload, fallback)
    return _dispatch_external_tool("semantic_search", req, context={"source": "tool_runtime.search_tool"})


def _tool_lookup(payload: dict[str, Any], fallback: Any) -> ToolResult:
    req = _coerce_query_payload(payload, fallback)
    return _dispatch_external_tool("lookup", req, context={"source": "tool_runtime.lookup_tool"})


def _tool_mcp(payload: dict[str, Any], fallback: Any) -> ToolResult:
    req: dict[str, Any]
    if isinstance(fallback, dict):
        req = dict(fallback)
        req.update(payload)
    else:
        req = dict(payload)
    if isinstance(fallback, str) and not req.get("tool"):
        req["tool"] = fallback
    if not req.get("tool"):
        req["tool"] = req.get("tool_name", "generic_tool")
    if "args" not in req:
        args = req.get("payload", {})
        req["args"] = args if isinstance(args, dict) else {"value": args}
    return _dispatch_external_tool("mcp_call", req, context={"source": "tool_runtime.mcp_tool"})


def _tool_filesystem(payload: dict[str, Any], fallback: Any) -> ToolResult:
    req: dict[str, Any]
    if isinstance(fallback, dict):
        req = dict(fallback)
        req.update(payload)
    else:
        req = dict(payload)
    op = str(req.get("operation") or req.get("action") or req.get("op") or "").strip().lower()
    if not op:
        if req.get("oldText") is not None or req.get("old") is not None:
            op = "edit"
        elif req.get("content") is not None or req.get("text") is not None:
            op = "write"
        elif req.get("pattern"):
            op = "glob"
        elif req.get("regex"):
            op = "grep"
        else:
            op = "read"

    if op in {"read", "cat", "view", "show", "read_file"}:
        return _tool_read(req, fallback)
    if op in {"write", "create", "create_file", "write_file", "put"}:
        return _tool_write(req, fallback)
    if op in {"append", "append_file"}:
        req["append"] = True
        return _tool_write(req, fallback)
    if op in {"edit", "replace", "patch", "modify"}:
        if req.get("oldText") is None and req.get("old") is None and req.get("find") is None:
            return _err("filesystem_tool edit requires oldText/old/find")
        return _tool_edit(req, fallback)
    if op in {"replace_in_file"}:
        if req.get("oldText") is None and req.get("old") is None and req.get("find") is None:
            return _err("replace_in_file requires oldText/old/find")
        if req.get("newText") is None and req.get("new") is None and req.get("replace") is None:
            return _err("replace_in_file requires newText/new/replace")
        return _tool_edit(req, fallback)
    if op in {"glob", "list", "ls", "list_files", "find_files"}:
        if not req.get("pattern"):
            req["pattern"] = req.get("glob") or req.get("filter") or "*"
        if not req.get("path"):
            req["path"] = "."
        return _tool_glob(req, fallback)
    if op in {"grep", "search_text", "find_text"}:
        if not req.get("pattern"):
            req["pattern"] = req.get("regex") or req.get("query") or req.get("text") or ""
        if not req.get("path"):
            req["path"] = "."
        return _tool_grep(req, fallback)
    return _err(f"filesystem_tool unsupported operation: {op}")


def _tool_calculate(payload: dict[str, Any], fallback: Any) -> ToolResult:
    expr = payload.get("expression") or payload.get("expr") or payload.get("input") or payload.get("text")
    if expr is None and isinstance(fallback, str):
        expr = fallback
    if expr is None and isinstance(fallback, dict):
        expr = fallback.get("expression") or fallback.get("expr") or fallback.get("input") or fallback.get("text")
    expr_text = str(expr or "").strip()
    if not expr_text:
        return _err("calculate_tool requires expression")

    import ast
    import math
    import operator as _op

    allowed_bin = {
        ast.Add: _op.add,
        ast.Sub: _op.sub,
        ast.Mult: _op.mul,
        ast.Div: _op.truediv,
        ast.FloorDiv: _op.floordiv,
        ast.Mod: _op.mod,
        ast.Pow: _op.pow,
    }
    allowed_unary = {
        ast.UAdd: _op.pos,
        ast.USub: _op.neg,
    }
    allowed_funcs = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
    }
    allowed_names = {
        "pi": math.pi,
        "e": math.e,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("Unsupported constant")
        if isinstance(node, ast.BinOp):
            fn = allowed_bin.get(type(node.op))
            if fn is None:
                raise ValueError("Unsupported binary operator")
            return float(fn(_eval(node.left), _eval(node.right)))
        if isinstance(node, ast.UnaryOp):
            fn = allowed_unary.get(type(node.op))
            if fn is None:
                raise ValueError("Unsupported unary operator")
            return float(fn(_eval(node.operand)))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Unsupported function call")
            fn = allowed_funcs.get(node.func.id)
            if fn is None:
                raise ValueError(f"Unsupported function: {node.func.id}")
            args = [_eval(arg) for arg in node.args]
            return float(fn(*args))
        if isinstance(node, ast.Name):
            if node.id in allowed_names:
                return float(allowed_names[node.id])
            raise ValueError(f"Unknown name: {node.id}")
        raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    try:
        parsed = ast.parse(expr_text, mode="eval")
        value = _eval(parsed)
    except Exception as exc:
        return _err(f"calculate_tool error: {type(exc).__name__}: {exc}")
    out = {"expression": expr_text, "result": value}
    return _ok(json.dumps(out, ensure_ascii=False), data=out)


def _tool_update_plan(payload: dict[str, Any], fallback: Any) -> ToolResult:
    plan = payload.get("plan")
    explanation = payload.get("explanation")
    if plan is None and isinstance(fallback, dict):
        plan = fallback.get("plan")
        explanation = fallback.get("explanation", explanation)
    if plan is None and isinstance(fallback, list):
        plan = fallback
    if plan is None and isinstance(fallback, str) and fallback.strip():
        try:
            parsed = json.loads(fallback)
            if isinstance(parsed, dict):
                plan = parsed.get("plan")
                explanation = parsed.get("explanation", explanation)
            elif isinstance(parsed, list):
                plan = parsed
        except json.JSONDecodeError:
            pass
    if plan is None:
        return _err("update_plan requires plan payload")
    _RUNTIME_STATE["update_plan"] = {"plan": plan, "explanation": explanation}
    return _ok(
        json.dumps({"plan": plan, "explanation": explanation}, ensure_ascii=False),
        data={"plan": plan, "explanation": explanation},
    )


def _tool_apply_patch(payload: dict[str, Any], fallback: Any) -> ToolResult:
    patch_text = payload.get("patch")
    if patch_text is None and isinstance(fallback, str):
        patch_text = fallback
    if not patch_text:
        return _err("apply_patch requires patch text")
    apply_patch_bin = shutil.which("apply_patch")
    if not apply_patch_bin:
        return _err("apply_patch binary not found in PATH")
    try:
        proc = subprocess.run(
            [apply_patch_bin],
            input=str(patch_text),
            text=True,
            capture_output=True,
            timeout=30,
        )
    except Exception as exc:
        return _err(f"apply_patch execution error: {type(exc).__name__}: {exc}")

    output = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    merged = output if not err else (output + ("\n" if output else "") + err)
    if proc.returncode != 0:
        return _err(merged or f"apply_patch failed with code {proc.returncode}")
    return _ok(merged or "apply_patch: success")


def _tool_parallel(payload: dict[str, Any], fallback: Any, *, _depth: int) -> ToolResult:
    if _depth >= 2:
        return _err("parallel nesting depth exceeded")
    calls = payload.get("tool_uses")
    if calls is None and isinstance(fallback, dict):
        calls = fallback.get("tool_uses")
    if not isinstance(calls, list):
        return _err("parallel requires tool_uses list")

    results: list[dict[str, Any]] = []
    for item in calls:
        if not isinstance(item, dict):
            results.append({"ok": False, "error": f"invalid tool use: {item!r}"})
            continue
        recipient = str(item.get("recipient_name") or item.get("tool_name") or "")
        if recipient.startswith("functions."):
            recipient = recipient.split(".", 1)[1]
        params = item.get("parameters", item.get("arguments", item.get("input", "")))
        if isinstance(params, (dict, list)):
            tool_input = json.dumps(params, ensure_ascii=False)
        else:
            tool_input = str(params)
        result = dispatch_tool_operation(recipient, tool_input, _depth=_depth + 1)
        results.append({"tool": recipient, "result": result})
    return _ok(json.dumps(results, ensure_ascii=False, indent=2), data={"results": results})


def _tool_rpc(payload: dict[str, Any], fallback: Any) -> ToolResult:
    method = payload.get("rpc_method") or payload.get("event_name") or (fallback if isinstance(fallback, str) else "rpc_event")
    ack = {"acknowledged": True, "method": str(method)}
    return _ok(json.dumps(ack, ensure_ascii=False), data=ack)


def _normalize_task_mode(raw: Any) -> str:
    mode = str(raw or os.environ.get("AGENT_ONTOLOGY_TASK_TOOL_MODE", "deterministic")).strip().lower()
    if mode in {"local", "local_run", "subprocess", "delegate_run"}:
        return "local_run"
    if mode in {"llm", "live", "external", "agent"}:
        return "llm"
    if mode in {"disabled", "off", "none"}:
        return "disabled"
    return "deterministic"


def _extract_task_request(payload: dict[str, Any], fallback: Any) -> dict[str, Any]:
    objective = (
        payload.get("objective")
        or payload.get("task")
        or payload.get("prompt")
        or payload.get("instruction")
        or payload.get("query")
        or payload.get("text")
    )
    if objective is None and isinstance(fallback, str):
        objective = fallback
    if objective is None and isinstance(fallback, dict):
        objective = (
            fallback.get("objective")
            or fallback.get("task")
            or fallback.get("prompt")
            or fallback.get("instruction")
            or fallback.get("query")
            or fallback.get("text")
        )
    objective_text = str(objective or "").strip()

    context = payload.get("context", payload.get("metadata", {}))
    if not isinstance(context, dict):
        context = {"value": context}
    acceptance = payload.get("acceptance_criteria", payload.get("criteria", []))
    if not isinstance(acceptance, list):
        acceptance = [acceptance] if acceptance is not None else []

    return {
        "objective": objective_text,
        "context": context,
        "acceptance_criteria": [str(item) for item in acceptance if item is not None],
        "requested_mode": str(payload.get("requested_mode", "default") or "default"),
    }


def _task_result_payload(*, delegate_id: str, mode: str, request: dict[str, Any], summary: str, done: bool) -> dict[str, Any]:
    context = request.get("context", {})
    context_keys = sorted(str(k) for k in context.keys()) if isinstance(context, dict) else []
    handoff = {
        "delegate_id": delegate_id,
        "objective": request.get("objective", ""),
        "context": context,
        "acceptance_criteria": request.get("acceptance_criteria", []),
        "requested_mode": request.get("requested_mode", "default"),
    }
    result = {
        "status": "completed" if done else "in_progress",
        "mode": mode,
        "summary": summary,
        "context_keys": context_keys,
        "files": [],
        "next_actions": [
            "execute delegated implementation steps",
            "verify acceptance criteria",
            "return final artifacts and diff summary",
        ],
        "done": done,
    }
    return {
        "delegate_id": delegate_id,
        "handoff": handoff,
        "result": result,
    }


def _is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _task_path_search_roots() -> list[pathlib.Path]:
    this_dir = pathlib.Path(__file__).resolve().parent
    cwd = pathlib.Path.cwd().resolve()
    roots = [
        cwd,
        this_dir,
        this_dir.parent,
        this_dir.parent.parent,
    ]
    out: list[pathlib.Path] = []
    seen: set[str] = set()
    for item in roots:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _resolve_existing_file(raw: Any, *, roots: list[pathlib.Path] | None = None) -> pathlib.Path | None:
    if not raw:
        return None
    path = pathlib.Path(str(raw)).expanduser()
    candidates: list[pathlib.Path] = []
    if path.is_absolute():
        candidates.append(path.resolve())
    else:
        search_roots = roots or _task_path_search_roots()
        for root in search_roots:
            try:
                candidates.append((root / path).resolve())
            except Exception:
                continue
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_output_path(raw: Any) -> pathlib.Path | None:
    if not raw:
        return None
    path = pathlib.Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = (pathlib.Path.cwd() / path).resolve()
    return path


def _resolve_task_spec_path(payload: dict[str, Any], request: dict[str, Any]) -> pathlib.Path | None:
    candidates: list[Any] = [
        payload.get("spec_path"),
        payload.get("delegate_spec_path"),
        payload.get("child_spec_path"),
        request.get("spec_path"),
    ]
    ctx = request.get("context")
    if isinstance(ctx, dict):
        candidates.extend(
            [
                ctx.get("spec_path"),
                ctx.get("delegate_spec_path"),
                ctx.get("child_spec_path"),
            ]
        )
    candidates.append(os.environ.get("AGENT_ONTOLOGY_TASK_TOOL_SPEC_PATH"))
    for raw in candidates:
        resolved = _resolve_existing_file(raw)
        if resolved is not None:
            return resolved
    return None


def _resolve_task_agent_path(payload: dict[str, Any], request: dict[str, Any]) -> pathlib.Path | None:
    candidates: list[Any] = [
        payload.get("agent_path"),
        payload.get("delegate_agent_path"),
        payload.get("sub_agent_path"),
        request.get("agent_path"),
    ]
    ctx = request.get("context")
    if isinstance(ctx, dict):
        candidates.extend(
            [
                ctx.get("agent_path"),
                ctx.get("delegate_agent_path"),
                ctx.get("sub_agent_path"),
            ]
        )
    candidates.append(os.environ.get("AGENT_ONTOLOGY_TASK_TOOL_AGENT_PATH"))

    for raw in candidates:
        resolved = _resolve_existing_file(raw)
        if resolved is not None:
            return resolved
    return None


def _resolve_instantiate_script() -> pathlib.Path | None:
    env_path = os.environ.get("AGENT_ONTOLOGY_TASK_TOOL_INSTANTIATE_SCRIPT")
    resolved_env = _resolve_existing_file(env_path)
    if resolved_env is not None:
        return resolved_env

    this_dir = pathlib.Path(__file__).resolve().parent
    candidates = [
        this_dir / "instantiate.py",
        this_dir.parent / "agent_ontology" / "instantiate.py",
        this_dir.parent / "instantiate.py",
    ]
    for root in _task_path_search_roots():
        candidates.append(root / "agent_ontology" / "instantiate.py")
        candidates.append(root / "instantiate.py")

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _derive_generated_agent_path(spec_path: pathlib.Path, payload: dict[str, Any]) -> pathlib.Path:
    explicit = (
        payload.get("generated_agent_path")
        or payload.get("delegate_output_agent_path")
        or payload.get("output_agent_path")
        or payload.get("agent_output_path")
    )
    explicit_path = _resolve_output_path(explicit)
    if explicit_path is not None:
        return explicit_path

    env_dir = os.environ.get("AGENT_ONTOLOGY_TASK_TOOL_GENERATED_DIR")
    if env_dir:
        base = pathlib.Path(env_dir).expanduser()
        if not base.is_absolute():
            base = (pathlib.Path.cwd() / base).resolve()
        return base / f"{spec_path.stem.replace('-', '_')}_agent.py"

    if spec_path.parent.name == "specs":
        return (spec_path.parent.parent / "agents" / f"{spec_path.stem.replace('-', '_')}_agent.py").resolve()

    digest = hashlib.sha1(str(spec_path).encode("utf-8")).hexdigest()[:8]
    fallback_dir = (pathlib.Path.cwd() / ".task_tool_agents").resolve()
    return fallback_dir / f"{spec_path.stem.replace('-', '_')}_{digest}_agent.py"


def _materialize_agent_from_spec(spec_path: pathlib.Path, payload: dict[str, Any]) -> tuple[pathlib.Path | None, dict[str, Any], str | None]:
    instantiate_script = _resolve_instantiate_script()
    if instantiate_script is None:
        return None, {}, (
            "task local_run could not find instantiate.py. "
            "Set AGENT_ONTOLOGY_TASK_TOOL_INSTANTIATE_SCRIPT to a valid path."
        )

    target_agent = _derive_generated_agent_path(spec_path, payload)
    force = _is_truthy(payload.get("regenerate")) or _is_truthy(payload.get("regenerate_agent")) or _is_truthy(
        payload.get("force_regenerate")
    )

    try:
        target_agent.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return None, {}, f"task local_run could not create generated agent dir: {type(exc).__name__}: {exc}"

    generated = True
    if target_agent.exists() and target_agent.is_file() and not force:
        try:
            target_mtime = target_agent.stat().st_mtime
            spec_mtime = spec_path.stat().st_mtime
            script_mtime = instantiate_script.stat().st_mtime
            if target_mtime >= spec_mtime and target_mtime >= script_mtime:
                generated = False
        except OSError:
            generated = True

    stdout_tail = ""
    stderr_tail = ""
    if generated:
        timeout_seconds = int(
            payload.get("instantiate_timeout_seconds")
            or os.environ.get("AGENT_ONTOLOGY_TASK_TOOL_INSTANTIATE_TIMEOUT_SECONDS", "120")
            or 120
        )
        try:
            proc = subprocess.run(
                [sys.executable, str(instantiate_script), str(spec_path), "-o", str(target_agent)],
                capture_output=True,
                text=True,
                timeout=max(10, timeout_seconds),
            )
        except subprocess.TimeoutExpired:
            return None, {}, f"task local_run instantiate timed out after {timeout_seconds}s for {spec_path}"
        except Exception as exc:
            return None, {}, f"task local_run instantiate failed: {type(exc).__name__}: {exc}"

        stdout_tail = str(proc.stdout or "")[-2500:]
        stderr_tail = str(proc.stderr or "")[-2500:]
        if proc.returncode != 0:
            details = (stderr_tail or stdout_tail or "unknown instantiate error").strip()
            return None, {}, f"task local_run instantiate failed (exit={proc.returncode}): {details}"

    if not target_agent.exists() or not target_agent.is_file():
        return None, {}, f"task local_run instantiate did not produce expected agent file: {target_agent}"

    meta = {
        "source": "spec_path",
        "spec_path": str(spec_path),
        "agent_path": str(target_agent),
        "instantiate_script": str(instantiate_script),
        "generated": generated,
        "instantiate_stdout_tail": stdout_tail,
        "instantiate_stderr_tail": stderr_tail,
    }
    return target_agent, meta, None


def _extract_delegate_trace_records(stdout: str, run_cwd: pathlib.Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in stdout.splitlines():
        m = _TRACE_WRITTEN_RE.match(line.strip())
        if not m:
            continue
        trace_ref = m.group("path").strip()
        trace_path = pathlib.Path(trace_ref)
        if not trace_path.is_absolute():
            trace_path = (run_cwd / trace_path).resolve()
        path_key = str(trace_path)
        if path_key in seen:
            continue
        seen.add(path_key)
        traces.append(
            {
                "path": path_key,
                "child_run_id": _infer_child_run_id_from_trace_path(trace_path),
                "total_calls": int(m.group("calls")),
                "total_duration_ms": int(m.group("dur")),
            }
        )
    return traces


def _emit_delegate_event(marker: str, payload: dict[str, Any]) -> None:
    if not _is_truthy(os.environ.get("AGENT_ONTOLOGY_EMIT_DELEGATE_EVENTS")):
        return
    try:
        print(marker + json.dumps(payload, ensure_ascii=False), flush=True)
    except Exception:
        return


def _extract_final_text(data: dict[str, Any]) -> str:
    for key in ("answer", "final_output", "draft_response", "response", "text", "action_result"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _task_via_local_run(delegate_id: str, request: dict[str, Any], payload: dict[str, Any]) -> ToolResult:
    agent_path = _resolve_task_agent_path(payload, request)
    materialization_meta: dict[str, Any] = {}
    if agent_path is None:
        spec_path = _resolve_task_spec_path(payload, request)
        if spec_path is not None:
            resolved_agent, meta, materialize_error = _materialize_agent_from_spec(spec_path, payload)
            if materialize_error:
                return _err(materialize_error, data={"spec_path": str(spec_path)})
            if resolved_agent is not None:
                agent_path = resolved_agent
            materialization_meta = meta

    if agent_path is None:
        return _err(
            "task local_run requires delegate agent path or spec_path. "
            "Set payload.agent_path, payload.spec_path, or AGENT_ONTOLOGY_TASK_TOOL_AGENT_PATH."
        )

    objective = str(request.get("objective", "") or "").strip()
    initial_data = {
        "text": objective,
        "task": objective,
        "objective": objective,
        "constraints": request.get("acceptance_criteria", []),
        "requested_mode": request.get("requested_mode", "default"),
        "context": request.get("context", {}),
    }
    extra_initial = payload.get("initial_data")
    if isinstance(extra_initial, dict):
        initial_data.update(extra_initial)

    timeout_seconds = int(
        payload.get("timeout_seconds")
        or os.environ.get("AGENT_ONTOLOGY_TASK_TOOL_TIMEOUT_SECONDS", "60")
        or 60
    )
    max_output_chars = int(
        payload.get("max_output_chars")
        or os.environ.get("AGENT_ONTOLOGY_TASK_TOOL_MAX_OUTPUT_CHARS", "16000")
        or 16000
    )
    run_cwd = pathlib.Path(_workdir_from_payload(payload)).expanduser()
    if not run_cwd.is_absolute():
        run_cwd = (pathlib.Path.cwd() / run_cwd).resolve()
    if not run_cwd.exists() or not run_cwd.is_dir():
        return _err(f"task local_run invalid workdir: {run_cwd}")

    runner = (
        "import importlib.util, json, pathlib, sys\n"
        "agent_path = pathlib.Path(sys.argv[1])\n"
        "initial = json.loads(sys.argv[2])\n"
        "spec = importlib.util.spec_from_file_location('delegate_agent_runtime', str(agent_path))\n"
        "if spec is None or spec.loader is None:\n"
        "    raise RuntimeError(f'Unable to load agent module: {agent_path}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "if not hasattr(mod, 'run'):\n"
        "    raise RuntimeError('Delegate agent module missing run(initial) function')\n"
        "state = mod.run(initial)\n"
        "if isinstance(state, dict):\n"
        "    final_data = state\n"
        "else:\n"
        "    final_data = getattr(state, 'data', {})\n"
        "if not isinstance(final_data, dict):\n"
        "    final_data = {'value': final_data}\n"
        "print('__AO_TASK_FINAL__' + json.dumps(final_data, default=str, ensure_ascii=False))\n"
    )

    env = os.environ.copy()
    if str(payload.get("allow_nested_task", False)).lower() not in {"1", "true", "yes", "on"}:
        env["AGENT_ONTOLOGY_TASK_TOOL_MODE"] = "disabled"

    t0 = time.time()
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                "-c",
                runner,
                str(agent_path),
                json.dumps(initial_data, ensure_ascii=False),
            ],
            capture_output=True,
            text=True,
            timeout=max(1, timeout_seconds),
            env=env,
            cwd=str(run_cwd),
        )
    except subprocess.TimeoutExpired:
        return _err(f"task local_run timed out after {timeout_seconds}s")
    except Exception as exc:
        return _err(f"task local_run execution error: {type(exc).__name__}: {exc}")
    duration_ms = int((time.time() - t0) * 1000)

    stdout = str(proc.stdout or "")
    stderr = str(proc.stderr or "")
    marker = "__AO_TASK_FINAL__"
    final_data: dict[str, Any] | None = None
    for line in stdout.splitlines():
        if line.startswith(marker):
            try:
                parsed = json.loads(line[len(marker) :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                final_data = parsed
    if proc.returncode != 0:
        details = (stderr.strip() or stdout.strip())[-1200:]
        return _err(
            f"task local_run failed (exit={proc.returncode}): {details}",
            data={
                "agent_path": str(agent_path),
                "returncode": proc.returncode,
                "duration_ms": duration_ms,
            },
        )
    if final_data is None:
        details = stdout.strip()[-1200:] or stderr.strip()[-1200:]
        return _err(
            f"task local_run completed but no final marker found. tail={details}",
            data={
                "agent_path": str(agent_path),
                "returncode": proc.returncode,
                "duration_ms": duration_ms,
            },
        )

    summary = _extract_final_text(final_data) or f"Delegated run completed via {agent_path.name}"
    done = bool(final_data.get("_done", False))
    trace_records = _extract_delegate_trace_records(stdout, run_cwd)
    payload_obj = _task_result_payload(
        delegate_id=delegate_id,
        mode="local_run",
        request=request,
        summary=summary,
        done=done,
    )
    result_data = payload_obj.get("result", {})
    if isinstance(result_data, dict):
        artifact_files = final_data.get("artifact_files")
        if isinstance(artifact_files, list):
            result_data["files"] = [str(item) for item in artifact_files]
    payload_obj["runtime"] = {
        "agent_path": str(agent_path),
        "returncode": proc.returncode,
        "duration_ms": duration_ms,
        "run_cwd": str(run_cwd),
        "delegate_trace_files": trace_records,
        "stdout_tail": stdout[-max_output_chars:],
        "stderr_tail": stderr[-2000:],
    }
    if materialization_meta:
        payload_obj["runtime"]["delegate_materialization"] = materialization_meta

    for record in trace_records:
        event_payload = {
            "delegate_id": delegate_id,
            "agent_path": str(agent_path),
            **record,
        }
        _emit_delegate_event("__AO_DELEGATE_TRACE__", event_payload)
    child_run_ids = sorted(
        {
            str(rec.get("child_run_id")).strip()
            for rec in trace_records
            if str(rec.get("child_run_id") or "").strip()
        }
    )
    child_trace_paths = [
        str(rec.get("path")).strip()
        for rec in trace_records
        if str(rec.get("path") or "").strip()
    ]
    _emit_delegate_event(
        "__AO_DELEGATE_RUN__",
        {
            "delegate_id": delegate_id,
            "agent_path": str(agent_path),
            "mode": "local_run",
            "returncode": proc.returncode,
            "duration_ms": duration_ms,
            "done": done,
            "trace_count": len(trace_records),
            "materialized_from_spec": bool(materialization_meta),
            "child_run_ids": child_run_ids,
            "child_run_id": child_run_ids[0] if len(child_run_ids) == 1 else "",
            "child_trace_paths": child_trace_paths,
        },
    )

    payload_obj["child_final_data_preview"] = {
        k: final_data[k]
        for k in list(final_data.keys())[:20]
    }
    _RUNTIME_STATE["task"][delegate_id] = payload_obj
    return _ok(json.dumps(payload_obj, ensure_ascii=False), data=payload_obj)


def _task_via_llm(delegate_id: str, request: dict[str, Any]) -> ToolResult:
    model = str(
        request.get("model")
        or os.environ.get("AGENT_ONTOLOGY_TASK_TOOL_MODEL", "openrouter/deepseek/deepseek-chat")
        or "openrouter/deepseek/deepseek-chat"
    ).strip()
    trace_id = f"agent_ontology/task_tool/{delegate_id}"
    system = (
        "You are a delegated sub-agent. Return strict JSON with keys: "
        "summary (string), done (boolean), next_actions (list<string>), files (list<string>)."
    )
    user = json.dumps(
        {
            "objective": request.get("objective", ""),
            "acceptance_criteria": request.get("acceptance_criteria", []),
            "context": request.get("context", {}),
        },
        ensure_ascii=False,
    )
    try:
        try:
            from llm_client import call_llm as llm_call
        except ImportError:
            from llm_client import call_llm as llm_call  # type: ignore[no-redef]
    except Exception as exc:
        return _err(f"task llm unavailable: {type(exc).__name__}: {exc}")

    try:
        llm_result = llm_call(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            task="agent_ontology.task_tool",
            trace_id=trace_id,
            max_budget=float(os.environ.get("AGENT_ONTOLOGY_MAX_BUDGET_USD", "0") or 0),
            max_tokens=int(os.environ.get("AGENT_ONTOLOGY_TASK_TOOL_MAX_TOKENS", "500") or 500),
            num_retries=1,
        )
    except Exception as exc:
        return _err(f"task llm call failed: {type(exc).__name__}: {exc}")

    raw = str(getattr(llm_result, "content", "") or "").strip()
    if not raw:
        return _err("task llm returned empty content")

    parsed: dict[str, Any] | None = None
    candidate = raw
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        maybe = json.loads(candidate)
        if isinstance(maybe, dict):
            parsed = maybe
    except json.JSONDecodeError:
        pass
    if parsed is None:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                maybe = json.loads(candidate[start : end + 1])
                if isinstance(maybe, dict):
                    parsed = maybe
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        payload = _task_result_payload(
            delegate_id=delegate_id,
            mode="llm",
            request=request,
            summary=raw,
            done=False,
        )
        _RUNTIME_STATE["task"][delegate_id] = payload
        return _ok(json.dumps(payload, ensure_ascii=False), data=payload)

    summary = str(parsed.get("summary", raw)).strip() or raw
    done = bool(parsed.get("done", False))
    payload = _task_result_payload(
        delegate_id=delegate_id,
        mode="llm",
        request=request,
        summary=summary,
        done=done,
    )
    result_data = payload["result"]
    if isinstance(result_data, dict):
        if isinstance(parsed.get("next_actions"), list):
            result_data["next_actions"] = [str(x) for x in parsed["next_actions"]]
        if isinstance(parsed.get("files"), list):
            result_data["files"] = [str(x) for x in parsed["files"]]
    _RUNTIME_STATE["task"][delegate_id] = payload
    return _ok(json.dumps(payload, ensure_ascii=False), data=payload)


def _tool_task(payload: dict[str, Any], fallback: Any) -> ToolResult:
    request = _extract_task_request(payload, fallback)
    objective = str(request.get("objective", "") or "").strip()
    if not objective:
        return _err("task requires objective/task text")

    mode = _normalize_task_mode(payload.get("mode"))
    delegate_id = str(payload.get("delegate_id") or f"delegate_{uuid.uuid4().hex[:12]}")
    request["delegate_id"] = delegate_id

    if mode == "disabled":
        return _err("task tool is disabled by AGENT_ONTOLOGY_TASK_TOOL_MODE")

    if mode == "local_run":
        return _task_via_local_run(delegate_id, request, payload)

    if mode == "llm":
        llm_result = _task_via_llm(delegate_id, request)
        if llm_result.get("ok"):
            return llm_result
        # Fall back to deterministic contract if llm delegation fails.

    summary = f"Delegated objective accepted: {objective}"
    payload_obj = _task_result_payload(
        delegate_id=delegate_id,
        mode="deterministic",
        request=request,
        summary=summary,
        done=False,
    )
    _RUNTIME_STATE["task"][delegate_id] = payload_obj
    return _ok(json.dumps(payload_obj, ensure_ascii=False), data=payload_obj)


def _tool_notebook_edit(payload: dict[str, Any], fallback: Any) -> ToolResult:
    merged = dict(payload)
    if isinstance(fallback, dict):
        for k, v in fallback.items():
            merged.setdefault(k, v)

    raw_path = merged.get("path") or merged.get("notebook_path")
    if not raw_path:
        return _err("NotebookEdit requires path")
    path = pathlib.Path(str(raw_path))
    create_if_missing = bool(merged.get("create", True))

    if path.exists():
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return _err(f"NotebookEdit read error for {path}: {type(exc).__name__}: {exc}")
        if not isinstance(notebook, dict):
            return _err(f"NotebookEdit expected notebook JSON object at {path}")
    else:
        if not create_if_missing:
            return _err(f"NotebookEdit path does not exist: {path}")
        notebook = {
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    cells = notebook.get("cells")
    if not isinstance(cells, list):
        cells = []
        notebook["cells"] = cells

    operation = str(merged.get("operation") or "").strip().lower()
    if operation not in {"append", "set", "replace_text", "append_cell", "set_cell"}:
        operation = "set"

    cell_type = str(merged.get("cell_type") or "code").strip().lower()
    if cell_type not in {"code", "markdown", "raw"}:
        cell_type = "code"

    source_value = (
        merged.get("new_source")
        or merged.get("source")
        or merged.get("text")
        or merged.get("content")
        or ""
    )
    source_text = str(source_value)
    source_lines = source_text.splitlines(keepends=True)
    if source_text and not source_lines:
        source_lines = [source_text]
    elif source_text and source_lines and not source_lines[-1].endswith("\n"):
        source_lines[-1] = source_lines[-1] + "\n"

    old_text = merged.get("old_text", merged.get("find"))
    new_text = merged.get("new_text", merged.get("replace"))
    try:
        cell_index = int(merged.get("cell_index", 0) or 0)
    except Exception:
        cell_index = 0
    cell_index = max(0, cell_index)

    def _mk_cell(src_lines: list[str], ctype: str) -> dict[str, Any]:
        base: dict[str, Any] = {
            "cell_type": ctype,
            "metadata": {},
            "source": list(src_lines),
        }
        if ctype == "code":
            base.setdefault("execution_count", None)
            base.setdefault("outputs", [])
        return base

    edited = False
    if operation in {"append", "append_cell"}:
        cells.append(_mk_cell(source_lines, cell_type))
        cell_index = len(cells) - 1
        edited = True
    elif operation in {"set", "set_cell"}:
        if cell_index >= len(cells):
            while len(cells) < cell_index:
                cells.append(_mk_cell([], "code"))
            cells.append(_mk_cell(source_lines, cell_type))
        else:
            cell = cells[cell_index]
            if not isinstance(cell, dict):
                cell = _mk_cell([], cell_type)
                cells[cell_index] = cell
            cell["cell_type"] = cell_type
            cell["source"] = list(source_lines)
            if cell_type == "code":
                cell.setdefault("execution_count", None)
                cell.setdefault("outputs", [])
        edited = True
    elif operation == "replace_text":
        if old_text is None or new_text is None:
            return _err("NotebookEdit replace_text requires old_text/new_text")
        old_s = str(old_text)
        new_s = str(new_text)
        replace_all = bool(merged.get("all", True))
        replaced_count = 0
        indices: list[int]
        if merged.get("cell_index") is not None:
            indices = [min(cell_index, max(0, len(cells) - 1))] if cells else []
        else:
            indices = list(range(len(cells)))
        for idx in indices:
            cell = cells[idx]
            if not isinstance(cell, dict):
                continue
            src = cell.get("source", [])
            if isinstance(src, list):
                src_text = "".join(str(x) for x in src)
            else:
                src_text = str(src)
            if old_s not in src_text:
                continue
            if replace_all:
                new_src = src_text.replace(old_s, new_s)
            else:
                new_src = src_text.replace(old_s, new_s, 1)
            if new_src != src_text:
                cell["source"] = new_src.splitlines(keepends=True)
                replaced_count += 1
                edited = True
                if not replace_all:
                    break
        if replaced_count == 0:
            return _err(f"NotebookEdit replace_text could not find target text in {path}")

    if not edited:
        return _err("NotebookEdit made no changes")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        return _err(f"NotebookEdit write error for {path}: {type(exc).__name__}: {exc}")

    msg = f"NotebookEdit {operation} succeeded at {path} (cell_index={cell_index}, total_cells={len(cells)})"
    return _ok(
        msg,
        data={
            "path": str(path),
            "operation": operation,
            "cell_index": cell_index,
            "total_cells": len(cells),
        },
    )


def _tool_environment(payload: dict[str, Any], fallback: Any) -> ToolResult:
    state = _RUNTIME_STATE["environment"]
    if not isinstance(state, dict) or not state:
        state = {
            "step": 0,
            "location": "spawn",
            "inventory": [],
            "last_action": "init",
            "observation": "Ready. You are at spawn.",
        }
        _RUNTIME_STATE["environment"] = state

    action = payload.get("action")
    if action is None and isinstance(fallback, str):
        action = fallback
    if action is None:
        action = "observe"
    action_text = str(action).strip()
    action_l = action_text.lower()

    if action_l in {"reset", "init", "initialize"}:
        state.update(
            {
                "step": 0,
                "location": "spawn",
                "inventory": [],
                "last_action": "reset",
                "observation": "Environment reset. You are at spawn.",
            }
        )
    elif action_l in {"observe", "status", "look"}:
        state["last_action"] = "observe"
    elif action_l.startswith("move "):
        dest = action_text.split(" ", 1)[1].strip() or "unknown"
        state["step"] = int(state.get("step", 0)) + 1
        state["location"] = dest
        state["last_action"] = action_text
        state["observation"] = f"Moved to {dest}."
    elif action_l.startswith("collect ") or action_l.startswith("pickup "):
        item = action_text.split(" ", 1)[1].strip() or "item"
        inv = state.get("inventory")
        if not isinstance(inv, list):
            inv = []
            state["inventory"] = inv
        inv.append(item)
        state["step"] = int(state.get("step", 0)) + 1
        state["last_action"] = action_text
        state["observation"] = f"Collected {item}."
    else:
        state["step"] = int(state.get("step", 0)) + 1
        state["last_action"] = action_text
        state["observation"] = f"Action acknowledged: {action_text}"

    result = {
        "acknowledged": True,
        "step": int(state.get("step", 0)),
        "location": str(state.get("location", "spawn")),
        "inventory": list(state.get("inventory", [])) if isinstance(state.get("inventory"), list) else [],
        "observation": str(state.get("observation", "")),
        "last_action": str(state.get("last_action", "")),
    }
    return _ok(json.dumps(result, ensure_ascii=False), data=result)


def _dynamic_filesystem_from_unknown(normalized: str, payload: dict[str, Any], fallback: Any) -> ToolResult:
    req = dict(payload)
    if isinstance(fallback, dict):
        merged = dict(fallback)
        merged.update(req)
        req = merged
    name = str(normalized or "")
    if not req.get("operation"):
        if any(tok in name for tok in {"read", "view", "cat", "show"}):
            req["operation"] = "read_file"
        elif any(tok in name for tok in {"write", "save", "create", "append"}):
            req["operation"] = "write_file"
        elif any(tok in name for tok in {"edit", "replace", "patch", "modify"}):
            req["operation"] = "replace_in_file"
        elif any(tok in name for tok in {"list", "find", "glob", "dir", "tree"}):
            req["operation"] = "list_files"
        elif any(tok in name for tok in {"grep", "search_text", "find_text", "match"}):
            req["operation"] = "search_text"
    if "replace" in str(req.get("operation", "")) and req.get("newText") is None and req.get("new") is None:
        if req.get("replacement") is not None:
            req["newText"] = req.get("replacement")
    if any(tok in name for tok in {"append"}) and req.get("operation") == "write_file":
        req["operation"] = "append_file"
    return _tool_filesystem(req, fallback)


def _dynamic_unknown_tool_dispatch(normalized: str, payload: dict[str, Any], fallback: Any) -> ToolResult:
    name = str(normalized or "").strip().lower()
    if not name:
        return _err("Unknown tool: <empty>", data={"tool_name": normalized})

    if any(tok in name for tok in {"parallel", "multi_tool_use", "fanout"}):
        return _tool_parallel(payload, fallback, _depth=0)
    if any(tok in name for tok in {"exec", "shell", "bash", "command", "terminal"}):
        return _tool_exec_command(payload, fallback)
    if "stdin" in name:
        return _tool_write_stdin(payload)
    if any(tok in name for tok in {"notebook", "ipynb"}):
        return _tool_notebook_edit(payload, fallback)
    if any(tok in name for tok in {"environment", "minecraft", "sim"}):
        return _tool_environment(payload, fallback)
    if any(tok in name for tok in {"task", "delegate", "sub_agent", "subagent"}):
        return _tool_task(payload, fallback)
    if any(tok in name for tok in {"update_plan", "todo_plan", "plan_update"}):
        return _tool_update_plan(payload, fallback)
    if "apply_patch" in name:
        return _tool_apply_patch(payload, fallback)
    if any(tok in name for tok in {"calc", "math", "evaluate_expression"}):
        return _tool_calculate(payload, fallback)
    if "rpc" in name or "emit" in name:
        return _tool_rpc(payload, fallback)
    if "mcp" in name:
        return _tool_mcp(payload, fallback)
    if any(tok in name for tok in {"search", "lookup", "retrieve", "semantic", "web", "browse"}):
        if str(payload.get("url", "")).strip().lower().startswith(("http://", "https://")):
            return _tool_web_fetch(payload, fallback)
        if "lookup" in name:
            return _tool_lookup(payload, fallback)
        return _tool_search(payload, fallback)
    if any(
        tok in name
        for tok in {
            "file",
            "filesystem",
            "fs",
            "path",
            "dir",
            "directory",
            "write",
            "read",
            "append",
            "replace",
            "edit",
            "glob",
            "grep",
        }
    ):
        return _dynamic_filesystem_from_unknown(name, payload, fallback)

    return _err(
        f"Unknown tool: {name}. No runtime adapter matched.",
        data={"tool_name": name, "payload_keys": sorted(payload.keys())},
    )


def dispatch_tool_operation(tool_name: str, input_text: str, *, _depth: int = 0) -> ToolResult:
    """Dispatch a tool call for generated agents."""

    normalized = _normalize_tool_name(tool_name)
    payload, fallback = _parse_input(input_text)

    aliases = {
        "bash": "exec_command",
        "exec_command": "exec_command",
        "exec_command_tool": "exec_command",
        "tool_exec_command_tool": "exec_command",
        "run_command": "exec_command",
        "run_shell_command": "exec_command",
        "terminal_exec": "exec_command",
        "shell_tool": "exec_command",
        "write_stdin": "write_stdin",
        "write_stdin_tool": "write_stdin",
        "tool_write_stdin_tool": "write_stdin",
        "stdin_write": "write_stdin",
        "read": "read",
        "write": "write",
        "edit": "edit",
        "glob": "glob",
        "grep": "grep",
        "filesystem_tool": "filesystem",
        "webfetch": "web_fetch",
        "web_fetch": "web_fetch",
        "web_tool": "web_fetch",
        "tool_web_tool": "web_fetch",
        "web_search": "search",
        "search_tool": "search",
        "semantic_search_tool": "search",
        "semantic_search": "search",
        "retrieve": "search",
        "retrieval": "search",
        "lookup": "lookup",
        "lookup_tool": "lookup",
        "mcp_call": "mcp",
        "mcp_tool_call": "mcp",
        "calculate_tool": "calculate",
        "calculate": "calculate",
        "math": "calculate",
        "update_plan": "update_plan",
        "update_plan_tool": "update_plan",
        "tool_update_plan_tool": "update_plan",
        "todo_update": "update_plan",
        "todo_plan_update": "update_plan",
        "apply_patch": "apply_patch",
        "apply_patch_tool": "apply_patch",
        "tool_apply_patch_tool": "apply_patch",
        "multi_tool_use.parallel": "parallel",
        "parallel": "parallel",
        "multi_tool_use": "parallel",
        "parallel_tool": "parallel",
        "tool_parallel_tool": "parallel",
        "rpc_emit": "rpc",
        "rpc_event": "rpc",
        "task_(sub_agent_spawn)": "task",
        "task": "task",
        "notebookedit": "notebook_edit",
        "notebook_edit": "notebook_edit",
        "rpc_tool": "rpc",
        "tool_rpc_tool": "rpc",
        "rpc": "rpc",
        "mcp_tool": "mcp",
        "mcp_tools": "mcp",
        "minecraft_environment": "environment",
        "environment": "environment",
    }
    op = aliases.get(normalized, normalized)

    # Common model-emitted filesystem aliases. Route these directly through
    # filesystem_tool semantics instead of failing as unknown tools.
    fs_alias_ops = {
        "read_file": "read_file",
        "write_file": "write_file",
        "create_file": "create_file",
        "append_file": "append_file",
        "replace_in_file": "replace_in_file",
        "edit_file": "edit",
        "list_files": "list_files",
        "find_files": "find_files",
        "search_text": "search_text",
    }
    if normalized in fs_alias_ops:
        if isinstance(payload, dict):
            payload = dict(payload)
            payload.setdefault("operation", fs_alias_ops[normalized])
        op = "filesystem"

    # write_code is a frequent alias from coding agents. Only treat it as a
    # filesystem write if the payload includes concrete path + code/content.
    if normalized == "write_code":
        req = dict(payload) if isinstance(payload, dict) else {}
        path = req.get("path") or req.get("file") or req.get("filename")
        content = req.get("content") or req.get("code")
        if path and content is not None:
            req["path"] = path
            req["content"] = content
            req.setdefault("operation", "write_file")
            payload = req
            op = "filesystem"
        else:
            return _err(
                "write_code requires path and content/code fields",
                data={"tool_name": tool_name, "payload_keys": sorted(req.keys())},
            )

    if op == "exec_command":
        return _tool_exec_command(payload, fallback)
    if op == "write_stdin":
        return _tool_write_stdin(payload)
    if op == "read":
        return _tool_read(payload, fallback)
    if op == "write":
        return _tool_write(payload, fallback)
    if op == "edit":
        return _tool_edit(payload, fallback)
    if op == "glob":
        return _tool_glob(payload, fallback)
    if op == "grep":
        return _tool_grep(payload, fallback)
    if op == "web_fetch":
        return _tool_web_fetch(payload, fallback)
    if op == "search":
        return _tool_search(payload, fallback)
    if op == "lookup":
        return _tool_lookup(payload, fallback)
    if op == "mcp":
        return _tool_mcp(payload, fallback)
    if op == "filesystem":
        return _tool_filesystem(payload, fallback)
    if op == "calculate":
        return _tool_calculate(payload, fallback)
    if op == "update_plan":
        return _tool_update_plan(payload, fallback)
    if op == "apply_patch":
        return _tool_apply_patch(payload, fallback)
    if op == "parallel":
        return _tool_parallel(payload, fallback, _depth=_depth)
    if op == "rpc":
        return _tool_rpc(payload, fallback)
    if op == "task":
        return _tool_task(payload, fallback)
    if op == "notebook_edit":
        return _tool_notebook_edit(payload, fallback)
    if op == "environment":
        return _tool_environment(payload, fallback)

    return _dynamic_unknown_tool_dispatch(normalized, payload, fallback)
