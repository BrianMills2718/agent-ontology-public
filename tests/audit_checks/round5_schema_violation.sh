#!/usr/bin/env bash
# Independent check for ChatGPT audit round 5 (schema-invalid LLM replies).
# Usage: tests/audit_checks/round5_schema_violation.sh <git-sha>
# Clones the repo fresh at <sha> into a new temp dir and runs agents/kb_react_agent.py
# with a scripted LLM whose first reply is valid JSON that violates ReActStep:
# {"type": "answer", "answer": 42}. Exits 0 only if run() fails loudly with an
# error naming ReActStep and the `answer` field, and the bad value was never merged
# (the step that would merge it is the one that raises). Exits 1 otherwise --
# e.g. the value is merged and emit_answer crashes on answer[:200] with TypeError,
# or the run completes as if the reply were fine.
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
llm_client.call_llm = lambda *a, **k: types.SimpleNamespace(
    content=json.dumps({"type": "answer", "thought": "done", "answer": 42}))
import kb_react_agent as m
state = m.AgentState()
state.data.update({"query": "q", "step_count": 0, "max_steps": 10, "trajectory": []})
try:
    m.process_think_or_act(state)
except Exception as e:
    merged = state.data.get("answer")
    print(f"raised {type(e).__name__}: {e}")
    print(f"state.data['answer'] after the step: {merged!r}")
    if type(e).__name__ == "SchemaViolationError" and "ReActStep" in str(e) and "answer" in str(e) and merged != 42:
        print("OK: schema-invalid reply refused loudly before merge")
        sys.exit(0)
    sys.exit("FAIL: wrong failure for a schema-invalid reply")
print(f"state.data['answer'] after the step: {state.data.get('answer')!r}")
try:
    m.process_emit_answer(state)
except TypeError as e:
    sys.exit(f"FAIL: invalid reply was merged; emit_answer crashed: TypeError: {e}")
sys.exit("FAIL: invalid reply was merged and the run carried on")
EOF
