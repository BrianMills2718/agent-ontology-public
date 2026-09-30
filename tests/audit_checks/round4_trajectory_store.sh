#!/usr/bin/env bash
# Independent check for ChatGPT audit rounds 3-4 (generated ReAct runtime).
# Usage: tests/audit_checks/round4_trajectory_store.sh <git-sha>
# Clones the repo fresh at <sha> into a new temp dir, runs agents/kb_react_agent.py
# with a scripted LLM (action given as tool_name/tool_input with no `action` text,
# then a thought, then an answer), and exits 0 only if:
#   - the thought step carries no observation (round 3),
#   - every persisted TrajectoryEntry equals the in-memory trajectory step (round 4),
#   - a provider failure propagates out of run() (round 3).
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
replies = [{"type": "action", "thought": "multiply", "tool_name": "calculate", "tool_input": "6*7"},
           {"type": "thought", "thought": "got it"}, {"type": "answer", "answer": "42"}]
q = [json.dumps(r) for r in replies]
llm_client.call_llm = lambda *a, **k: types.SimpleNamespace(content=q.pop(0))
import kb_react_agent as m
state = m.run({"query": "q"})
traj, stored = state.data["trajectory"], state.trajectory_store.queue
print("trajectory:", traj)
print("persisted: ", stored)
fails = []
if [t["observation"] for t in traj] != ["42", ""]:
    fails.append("thought-only step inherited the previous action's observation")
if stored != traj:
    fails.append("persisted TrajectoryEntry records disagree with the in-memory trajectory")
def down(*a, **k): raise RuntimeError("provider down")
llm_client.call_llm = down
try:
    m.run({"query": "q"})
    fails.append("provider failure swallowed; run() returned normally")
except RuntimeError:
    pass
if fails:
    sys.exit("FAIL: " + "; ".join(fails))
print("OK: observation cleared, persisted == trajectory, provider failure propagated")
EOF
