#!/usr/bin/env bash
# Independent check for ChatGPT audit round 6 (non-JSON LLM replies).
# Usage: tests/audit_checks/round7_unparseable_reply.sh <git-sha>
# Clones the repo fresh at <sha> into a new temp dir. It runs agents/kb_react_agent.py
# with an LLM that replies in prose (not JSON). Exits 0 only if run() raises
# SchemaViolationError naming ReActStep and quoting the reply. It also checks the
# free-text exception: minimal_solver, whose only output field is the string `answer`,
# must keep a prose reply as its answer. Exits 1 if the prose is swallowed into an
# empty thought and the run ends "No answer found" (the round-6 finding).
# Needs: git SSH access to the repo, python3 with pyyaml and llm_client importable.
set -u
sha="$1"
repo="${AO_REPO:-git@github-personal:BrianMills2718/agent_ontology.git}"
work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
git clone -q "$repo" "$work/ao" && git -C "$work/ao" checkout -q "$sha" || exit 2
cd "$work/ao" && python3 - <<'EOF'
import sys, types, llm_client
sys.path[:0] = ["agents", "."]
prose = "I think the answer is 42, because six times seven is forty-two."
llm_client.call_llm = lambda *a, **k: types.SimpleNamespace(content=prose)
import kb_react_agent as kb, minimal_solver_agent as ms
fails = []
try:
    s = kb.run({"query": "What is 6*7?"})
    print("kb_react returned normally; answer =", repr(s.data.get("answer")))
    fails.append("kb_react swallowed a non-JSON reply")
except Exception as e:
    print(f"kb_react raised {type(e).__name__}: {e}")
    if not (type(e).__name__ == "SchemaViolationError" and "ReActStep" in str(e) and prose[:30] in str(e)):
        fails.append("kb_react failed, but not with a SchemaViolationError quoting the reply")
s = ms.run({"problem": "What is 6*7?"})
print("minimal_solver answer =", repr(s.data.get("answer")))
if s.data.get("answer") != prose:
    fails.append("single-string-field schema did not keep the prose reply as its answer")
if fails:
    sys.exit("FAIL: " + "; ".join(fails))
print("OK: non-JSON reply refused loudly; free-text schema keeps prose")
EOF
