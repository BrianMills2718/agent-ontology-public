#!/usr/bin/env bash
# Independent check for the spawn/placeholder follow-up to ChatGPT audit round 5.
# Usage: tests/audit_checks/round6_spawn_placeholder.sh <git-sha>
# Clones the repo fresh at <sha> into a new temp dir and runs agents/map_reduce_agent.py's
# spawn step (map_chunks) directly. Exits 0 only if it raises NotImplementedError naming
# the agent and the process. Exits 1 if it prints a placeholder ("Would create
# sub-agent") and returns normally, which is what let map_reduce report success
# without ever mapping.
# Needs: git SSH access to the repo, python3 with pyyaml and llm_client importable.
set -u
sha="$1"
repo="${AO_REPO:-git@github-personal:BrianMills2718/agent_ontology.git}"
work="$(mktemp -d)"
trap 'rm -rf -- "$work"' EXIT
git clone -q "$repo" "$work/ao" && git -C "$work/ao" checkout -q "$sha" || exit 2
cd "$work/ao" && python3 - <<'EOF'
import sys
sys.path[:0] = ["agents", "."]
import map_reduce_agent as m
state = m.AgentState()
state.data.update({"document": "a b c", "chunks": [{"text": "a b c"}]})
try:
    m.process_map_chunks(state)
except NotImplementedError as e:
    print(f"raised NotImplementedError: {e}")
    if "map_chunks" in str(e) and "MapReduce Document Processor" in str(e):
        print("OK: unimplemented spawn fails loudly")
        sys.exit(0)
    sys.exit("FAIL: error does not name the process")
print("returned normally; chunk_result =", state.data.get("chunk_result"))
sys.exit("FAIL: spawn step printed a placeholder and continued; nothing was mapped")
EOF
