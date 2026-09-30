from __future__ import annotations

import json
import pytest
import shutil
from pathlib import Path

from agent_ontology.tool_runtime import dispatch_tool_operation


def test_file_tools_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"

    write_res = dispatch_tool_operation(
        "write",
        json.dumps({"path": str(target), "content": "alpha\nbeta\n"}, ensure_ascii=False),
    )
    assert write_res.get("handled") is True
    assert write_res.get("ok") is True
    assert target.exists()

    read_res = dispatch_tool_operation("read", json.dumps({"path": str(target)}, ensure_ascii=False))
    assert read_res.get("ok") is True
    assert "alpha" in str(read_res.get("output", ""))

    edit_res = dispatch_tool_operation(
        "edit",
        json.dumps({"path": str(target), "oldText": "beta", "newText": "gamma"}, ensure_ascii=False),
    )
    assert edit_res.get("ok") is True
    assert "gamma" in target.read_text(encoding="utf-8")

    glob_res = dispatch_tool_operation(
        "glob",
        json.dumps({"path": str(tmp_path), "pattern": "*.txt"}, ensure_ascii=False),
    )
    assert glob_res.get("ok") is True
    assert "notes.txt" in str(glob_res.get("output", ""))

    grep_res = dispatch_tool_operation(
        "grep",
        json.dumps({"path": str(tmp_path), "pattern": "gamma"}, ensure_ascii=False),
    )
    assert grep_res.get("ok") is True
    assert "gamma" in str(grep_res.get("output", ""))


@pytest.mark.skipif(
    shutil.which("apply_patch") is None,
    reason="apply_patch binary not in PATH; this exercises the external tool, not our code",
)
def test_apply_patch_add_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    patch_text = (
        "*** Begin Patch\n"
        "*** Add File: hello.txt\n"
        "+hello from patch\n"
        "*** End Patch\n"
    )
    result = dispatch_tool_operation("apply_patch", patch_text)
    assert result.get("handled") is True
    assert result.get("ok") is True
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello from patch\n"


def test_parallel_dispatch_exec_command_and_update_plan() -> None:
    payload = {
        "tool_uses": [
            {
                "recipient_name": "functions.exec_command",
                "parameters": {"cmd": "echo hello"},
            },
            {
                "recipient_name": "functions.update_plan",
                "parameters": {
                    "explanation": "Start work",
                    "plan": [{"step": "build runtime", "status": "in_progress"}],
                },
            },
        ]
    }
    result = dispatch_tool_operation("multi_tool_use.parallel", json.dumps(payload, ensure_ascii=False))
    assert result.get("handled") is True
    assert result.get("ok") is True
    text = str(result.get("output", ""))
    assert "exec_command" in text
    assert "update_plan" in text


def test_write_stdin_unknown_session() -> None:
    result = dispatch_tool_operation(
        "write_stdin",
        json.dumps({"session_id": 999999, "chars": "test"}, ensure_ascii=False),
    )
    assert result.get("handled") is True
    assert result.get("ok") is False
    assert "Unknown session_id" in str(result.get("output", ""))


def test_web_tool_query_uses_external_search_adapter() -> None:
    result = dispatch_tool_operation("web_tool", "state diff telemetry")
    assert result.get("handled") is True
    assert result.get("ok") is True
    text = str(result.get("output", ""))
    assert text


def test_task_tool_returns_typed_handoff_contract() -> None:
    payload = {
        "task": "build todo backend",
        "context": {"repo": "agent_ontology", "priority": "high"},
        "acceptance_criteria": ["create endpoints", "add tests"],
    }
    result = dispatch_tool_operation("Task (Sub-Agent Spawn)", json.dumps(payload, ensure_ascii=False))
    assert result.get("handled") is True
    assert result.get("ok") is True
    data = result.get("data")
    assert isinstance(data, dict)
    assert isinstance(data.get("delegate_id"), str)
    handoff = data.get("handoff")
    assert isinstance(handoff, dict)
    assert handoff.get("objective") == "build todo backend"
    contract_result = data.get("result")
    assert isinstance(contract_result, dict)
    assert contract_result.get("mode") in {"deterministic", "llm"}
    assert "summary" in contract_result


def test_task_tool_local_run_executes_delegate_agent(tmp_path: Path) -> None:
    delegate_path = tmp_path / "delegate_agent.py"
    delegate_path.write_text(
        "\n".join(
            [
                "class _State:",
                "    def __init__(self):",
                "        self.data = {}",
                "",
                "def run(initial=None):",
                "    s = _State()",
                "    if isinstance(initial, dict):",
                "        s.data.update(initial)",
                "    obj = str(s.data.get('objective', ''))",
                "    s.data['answer'] = f'delegated:{obj}'",
                "    s.data['_done'] = True",
                "    return s",
            ]
        ),
        encoding="utf-8",
    )

    payload = {
        "task": "implement todo routes",
        "mode": "local_run",
        "agent_path": str(delegate_path),
        "context": {"repo": "agent_ontology"},
    }
    result = dispatch_tool_operation("task", json.dumps(payload, ensure_ascii=False))
    assert result.get("handled") is True
    assert result.get("ok") is True
    data = result.get("data")
    assert isinstance(data, dict)
    runtime = data.get("runtime")
    assert isinstance(runtime, dict)
    assert runtime.get("returncode") == 0
    out = data.get("result")
    assert isinstance(out, dict)
    assert out.get("mode") == "local_run"
    assert out.get("done") is True
    assert "delegated:implement todo routes" in str(out.get("summary", ""))


def test_task_tool_local_run_requires_agent_path(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_ONTOLOGY_TASK_TOOL_AGENT_PATH", raising=False)
    payload = {"task": "implement todo routes", "mode": "local_run"}
    result = dispatch_tool_operation("task", json.dumps(payload, ensure_ascii=False))
    assert result.get("handled") is True
    assert result.get("ok") is False
    assert "requires delegate agent path" in str(result.get("output", ""))


def test_task_tool_local_run_accepts_spec_path_and_materializes_agent(tmp_path: Path, monkeypatch) -> None:
    spec_path = tmp_path / "child_delegate.yaml"
    spec_path.write_text("name: child\nversion: 0.1\n", encoding="utf-8")

    fake_instantiate = tmp_path / "fake_instantiate.py"
    fake_instantiate.write_text(
        "\n".join(
            [
                "import pathlib, sys",
                "spec = pathlib.Path(sys.argv[1])",
                "out = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])",
                "out.parent.mkdir(parents=True, exist_ok=True)",
                "out.write_text(",
                "    \"class _S:\\n\"",
                "    \"    def __init__(self):\\n\"",
                "    \"        self.data = {}\\n\\n\"",
                "    \"def run(initial=None):\\n\"",
                "    \"    s = _S()\\n\"",
                "    \"    if isinstance(initial, dict):\\n\"",
                "    \"        s.data.update(initial)\\n\"",
                "    \"    s.data['answer'] = 'from-spec:' + str(s.data.get('objective', ''))\\n\"",
                "    \"    s.data['_done'] = True\\n\"",
                "    \"    return s\\n\",",
                "    encoding='utf-8'",
                ")",
                "print(f'generated {out} from {spec}')",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AGENT_ONTOLOGY_TASK_TOOL_INSTANTIATE_SCRIPT", str(fake_instantiate))
    payload = {
        "task": "implement task tool",
        "mode": "local_run",
        "spec_path": str(spec_path),
        "generated_agent_path": str(tmp_path / "generated_delegate.py"),
        "regenerate_agent": True,
    }
    result = dispatch_tool_operation("task", json.dumps(payload, ensure_ascii=False))
    assert result.get("handled") is True
    assert result.get("ok") is True
    data = result.get("data")
    assert isinstance(data, dict)
    runtime = data.get("runtime")
    assert isinstance(runtime, dict)
    material = runtime.get("delegate_materialization")
    assert isinstance(material, dict)
    assert material.get("source") == "spec_path"
    agent_path = material.get("agent_path")
    assert isinstance(agent_path, str) and agent_path.endswith("generated_delegate.py")
    assert Path(agent_path).exists()
    out = data.get("result")
    assert isinstance(out, dict)
    assert out.get("done") is True
    assert "from-spec:implement task tool" in str(out.get("summary", ""))


def test_task_tool_local_run_extracts_delegate_trace_and_emits_events(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    delegate_path = tmp_path / "delegate_trace_agent.py"
    delegate_path.write_text(
        "\n".join(
            [
                "class _State:",
                "    def __init__(self):",
                "        self.data = {}",
                "",
                "def run(initial=None):",
                "    s = _State()",
                "    if isinstance(initial, dict):",
                "        s.data.update(initial)",
                "    print('Trace written to .live_runs/childrun123/trace.json (3 calls, 42ms total)')",
                "    s.data['answer'] = 'trace-ready'",
                "    s.data['_done'] = True",
                "    return s",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENT_ONTOLOGY_EMIT_DELEGATE_EVENTS", "1")
    payload = {
        "task": "trace behavior",
        "mode": "local_run",
        "agent_path": str(delegate_path),
        "workdir": str(tmp_path),
    }
    result = dispatch_tool_operation("task", json.dumps(payload, ensure_ascii=False))
    assert result.get("ok") is True
    runtime = result.get("data", {}).get("runtime", {})
    traces = runtime.get("delegate_trace_files")
    assert isinstance(traces, list) and traces
    first = traces[0]
    assert isinstance(first, dict)
    assert first.get("total_calls") == 3
    assert first.get("total_duration_ms") == 42
    assert first.get("child_run_id") == "childrun123"
    path = first.get("path")
    assert isinstance(path, str) and path.endswith("trace.json")

    stdout = capsys.readouterr().out
    assert "__AO_DELEGATE_TRACE__" in stdout
    assert "__AO_DELEGATE_RUN__" in stdout
    trace_lines = [line for line in stdout.splitlines() if line.startswith("__AO_DELEGATE_TRACE__")]
    run_lines = [line for line in stdout.splitlines() if line.startswith("__AO_DELEGATE_RUN__")]
    assert trace_lines and run_lines
    trace_payload = json.loads(trace_lines[-1].split("__AO_DELEGATE_TRACE__", 1)[1])
    run_payload = json.loads(run_lines[-1].split("__AO_DELEGATE_RUN__", 1)[1])
    assert trace_payload.get("child_run_id") == "childrun123"
    assert run_payload.get("child_run_id") == "childrun123"
    assert run_payload.get("child_run_ids") == ["childrun123"]


def test_notebook_edit_set_and_replace_text(tmp_path: Path) -> None:
    nb_path = tmp_path / "demo.ipynb"
    set_payload = {
        "path": str(nb_path),
        "operation": "set",
        "cell_index": 0,
        "cell_type": "markdown",
        "new_source": "hello notebook",
    }
    set_res = dispatch_tool_operation("NotebookEdit", json.dumps(set_payload, ensure_ascii=False))
    assert set_res.get("handled") is True
    assert set_res.get("ok") is True

    replace_payload = {
        "path": str(nb_path),
        "operation": "replace_text",
        "old_text": "hello",
        "new_text": "hi",
    }
    repl_res = dispatch_tool_operation("notebook_edit", json.dumps(replace_payload, ensure_ascii=False))
    assert repl_res.get("ok") is True

    nb_obj = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb_obj.get("cells", [])
    assert isinstance(cells, list) and cells
    src = "".join(cells[0].get("source", []))
    assert "hi notebook" in src


def test_environment_adapter_tracks_state() -> None:
    reset = dispatch_tool_operation("Minecraft Environment", json.dumps({"action": "reset"}, ensure_ascii=False))
    assert reset.get("ok") is True
    move = dispatch_tool_operation("environment", "move village")
    assert move.get("ok") is True
    data = move.get("data")
    assert isinstance(data, dict)
    assert data.get("location") == "village"
    collect = dispatch_tool_operation("environment", "collect wood")
    assert collect.get("ok") is True
    cdata = collect.get("data")
    assert isinstance(cdata, dict)
    assert "wood" in cdata.get("inventory", [])


def test_search_lookup_and_web_aliases_use_external_adapter() -> None:
    search_res = dispatch_tool_operation(
        "search_tool",
        json.dumps({"query": "livekit lifecycle"}, ensure_ascii=False),
    )
    assert search_res.get("handled") is True
    assert search_res.get("ok") is True
    assert str(search_res.get("output", ""))

    lookup_res = dispatch_tool_operation(
        "lookup_tool",
        json.dumps({"query": "rpc event handling"}, ensure_ascii=False),
    )
    assert lookup_res.get("handled") is True
    assert lookup_res.get("ok") is True
    assert str(lookup_res.get("output", ""))

    web_search_res = dispatch_tool_operation("web_search", "semantic retrieval")
    assert web_search_res.get("handled") is True
    assert web_search_res.get("ok") is True
    assert str(web_search_res.get("output", ""))


def test_mcp_tool_aliases_dispatch_through_external_adapter() -> None:
    payload = {
        "tool": "semantic_lookup",
        "args": {"query": "todo architecture"},
    }
    res = dispatch_tool_operation("mcp_tools", json.dumps(payload, ensure_ascii=False))
    assert res.get("handled") is True
    assert res.get("ok") is True
    data = res.get("data")
    assert isinstance(data, dict)
    assert data.get("interface_name") == "mcp_call"
    assert data.get("handled") is True


def test_calculate_tool_evaluates_expression_safely() -> None:
    res = dispatch_tool_operation("calculate_tool", json.dumps({"expression": "2 + 3 * (4 - 1)"}, ensure_ascii=False))
    assert res.get("handled") is True
    assert res.get("ok") is True
    data = res.get("data")
    assert isinstance(data, dict)
    assert data.get("result") == 11.0

    bad = dispatch_tool_operation("calculate_tool", json.dumps({"expression": "__import__('os').system('id')"}))
    assert bad.get("handled") is True
    assert bad.get("ok") is False


def test_filesystem_tool_supports_write_and_read(tmp_path: Path) -> None:
    target = tmp_path / "notes.md"
    write_payload = {
        "operation": "write",
        "path": str(target),
        "content": "# hello\n",
    }
    wr = dispatch_tool_operation("filesystem_tool", json.dumps(write_payload, ensure_ascii=False))
    assert wr.get("handled") is True
    assert wr.get("ok") is True
    assert target.exists()

    read_payload = {
        "operation": "read",
        "path": str(target),
    }
    rd = dispatch_tool_operation("filesystem_tool", json.dumps(read_payload, ensure_ascii=False))
    assert rd.get("ok") is True
    assert "# hello" in str(rd.get("output", ""))


def test_filesystem_alias_write_file_and_read_file(tmp_path: Path) -> None:
    target = tmp_path / "alias.txt"
    wr = dispatch_tool_operation(
        "write_file",
        json.dumps({"path": str(target), "content": "alias-ok\n"}, ensure_ascii=False),
    )
    assert wr.get("handled") is True
    assert wr.get("ok") is True
    assert target.exists()

    rd = dispatch_tool_operation("read_file", json.dumps({"path": str(target)}, ensure_ascii=False))
    assert rd.get("handled") is True
    assert rd.get("ok") is True
    assert "alias-ok" in str(rd.get("output", ""))


def test_filesystem_alias_create_and_append_file(tmp_path: Path) -> None:
    target = tmp_path / "append_alias.txt"
    create = dispatch_tool_operation(
        "create_file",
        json.dumps({"path": str(target), "content": "line-1\n"}, ensure_ascii=False),
    )
    assert create.get("handled") is True
    assert create.get("ok") is True
    assert target.read_text(encoding="utf-8") == "line-1\n"

    append = dispatch_tool_operation(
        "append_file",
        json.dumps({"path": str(target), "content": "line-2\n"}, ensure_ascii=False),
    )
    assert append.get("handled") is True
    assert append.get("ok") is True
    assert target.read_text(encoding="utf-8") == "line-1\nline-2\n"


def test_filesystem_alias_replace_in_file(tmp_path: Path) -> None:
    target = tmp_path / "replace_alias.txt"
    target.write_text("hello world\n", encoding="utf-8")
    repl = dispatch_tool_operation(
        "replace_in_file",
        json.dumps({"path": str(target), "oldText": "world", "newText": "agent"}, ensure_ascii=False),
    )
    assert repl.get("handled") is True
    assert repl.get("ok") is True
    assert target.read_text(encoding="utf-8") == "hello agent\n"


def test_write_file_alias_accepts_file_path_key(tmp_path: Path) -> None:
    target = tmp_path / "file_path_alias.txt"
    wr = dispatch_tool_operation(
        "write_file",
        json.dumps({"file_path": str(target), "content": "from-file-path\n"}, ensure_ascii=False),
    )
    assert wr.get("handled") is True
    assert wr.get("ok") is True
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "from-file-path\n"


def test_write_file_alias_accepts_filename_and_contents_keys(tmp_path: Path) -> None:
    target = tmp_path / "contents_alias.txt"
    wr = dispatch_tool_operation(
        "write_file",
        json.dumps({"filename": str(target), "contents": "from-contents\n"}, ensure_ascii=False),
    )
    assert wr.get("handled") is True
    assert wr.get("ok") is True
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "from-contents\n"


def test_write_code_alias_writes_when_path_and_content_present(tmp_path: Path) -> None:
    target = tmp_path / "calc.py"
    res = dispatch_tool_operation(
        "write_code",
        json.dumps({"path": str(target), "content": "print('calculator')\n"}, ensure_ascii=False),
    )
    assert res.get("handled") is True
    assert res.get("ok") is True
    assert target.read_text(encoding="utf-8") == "print('calculator')\n"


def test_write_code_alias_requires_path_and_content() -> None:
    res = dispatch_tool_operation("write_code", json.dumps({"language": "python"}, ensure_ascii=False))
    assert res.get("handled") is True
    assert res.get("ok") is False
    assert "requires path and content" in str(res.get("output", ""))


def test_shell_aliases_route_to_exec_command() -> None:
    bash_res = dispatch_tool_operation("bash", json.dumps({"cmd": "echo alias-ok"}, ensure_ascii=False))
    assert bash_res.get("handled") is True
    assert bash_res.get("ok") is True

    shell_res = dispatch_tool_operation("shell_tool", json.dumps({"cmd": "echo shell-ok"}, ensure_ascii=False))
    assert shell_res.get("handled") is True
    assert shell_res.get("ok") is True


def test_unknown_tool_name_with_retrieval_semantics_routes_to_search_adapter() -> None:
    res = dispatch_tool_operation(
        "retrieve_context_snippets",
        json.dumps({"query": "todo api pagination examples"}, ensure_ascii=False),
    )
    assert res.get("handled") is True
    assert res.get("ok") is True
    assert str(res.get("output", ""))


def test_unknown_tool_name_with_filesystem_semantics_routes_to_filesystem(tmp_path: Path) -> None:
    target = tmp_path / "auto.txt"
    res = dispatch_tool_operation(
        "save_file_text",
        json.dumps({"path": str(target), "content": "dynamic-dispatch\n"}, ensure_ascii=False),
    )
    assert res.get("handled") is True
    assert res.get("ok") is True
    assert target.read_text(encoding="utf-8") == "dynamic-dispatch\n"


def test_unknown_tool_name_returns_handled_error_without_not_implemented_stub() -> None:
    res = dispatch_tool_operation(
        "nonexistent_totally_custom_tool",
        json.dumps({"foo": "bar"}, ensure_ascii=False),
    )
    assert res.get("handled") is True
    assert res.get("ok") is False
    text = str(res.get("output", ""))
    assert "Unknown tool:" in text
    assert "not implemented" not in text.lower()
