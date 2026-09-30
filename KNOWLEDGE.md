# Operational Knowledge — agent_ontology

Shared findings from all agent sessions. Any agent brain can read and append.
Human-reviewed periodically.

## Findings

<!-- Append new findings below this line. Do not overwrite existing entries. -->
<!-- Format: ### YYYY-MM-DD — {agent} — {category}                          -->
<!-- Categories: bug-pattern, performance, schema-gotcha, integration-issue, -->
<!--             workaround, best-practice                                   -->
<!-- Agent names: claude-code, codex, openclaw                               -->

### 2026-04-04 — codex — integration-issue

Generated agents and runtime helpers should import `call_llm` / `embed` from the
public `llm_client` package surface, not `llm_client.llm_client`.

The nested import path is a monorepo-layout assumption that breaks once
`llm_client` is installed or exposed through its normal package facade. The
affected seams here were:

- `agent_ontology/external_interfaces.py`
- `agent_ontology/tool_runtime.py`
- generated agent code emitted by `instantiate.py`

Verification on the repaired surface:

- `PYTHONPATH=. pytest -q tests/test_external_interfaces.py tests/test_tool_runtime.py tests/test_agents.py`
