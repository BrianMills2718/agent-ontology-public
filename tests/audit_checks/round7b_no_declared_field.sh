#!/usr/bin/env bash
# Independent check: a JSON reply carrying none of the schema's declared fields.
# Usage: tests/audit_checks/round7b_no_declared_field.sh <git-sha>
# Clones the repo fresh at <sha> into a new temp dir and runs agents/kb_react_agent.py
# with an LLM that replies {"note": "..."}: valid JSON, but none of ReActStep's fields.
# Exits 0 only if run() raises SchemaViolationError naming ReActStep and quoting the
# reply, and BabyAGI still accepts a reply that carries only its optional
# `tool_calls` field. Exits 1 if the object is reset to defaults and the run
# carries on to "No answer found".
# Needs: git SSH access to the repo, python3 with pyyaml and llm_client importable.
set -u
sha="$1"
repo="${AO_REPO:-git@github-personal:BrianMills2718/agent_ontology.git}"
work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
git clone -q "$repo" "$work/ao" && git -C "$work/ao" checkout -q "$sha" || exit 2
cd "$work/ao" && python3 - <<'EOF'
import sys, json, types, llm_client
sys.path[:0] = ["agents", "."]
reply = json.dumps({"note": "I could not decide"})
llm_client.call_llm = lambda *a, **k: types.SimpleNamespace(content=reply)
import kb_react_agent as kb, babyagi_agent as ba
fails = []
try:
    s = kb.run({"query": "What is 6*7?"})
    print("kb_react returned normally; answer =", repr(s.data.get("answer")))
    fails.append("an object with no ReActStep field was accepted")
except Exception as e:
    print(f"kb_react raised {type(e).__name__}: {e}")
    if not (type(e).__name__ == "SchemaViolationError" and "ReActStep" in str(e) and "I could not decide" in str(e)):
        fails.append("kb_react failed, but not with a SchemaViolationError quoting the reply")
if hasattr(ba, "require_valid_output"):
    kept = ba.require_valid_output({"tool_calls": []}, "ExecutionOutput", "Execution Agent")
    print("babyagi tool_calls-only reply ->", kept)
    if kept != {"tool_calls": []}:
        fails.append("babyagi's optional-only reply was not kept")
if fails:
    sys.exit("FAIL: " + "; ".join(fails))
print("OK: object with no declared field refused loudly; optional-only reply still accepted")
EOF
