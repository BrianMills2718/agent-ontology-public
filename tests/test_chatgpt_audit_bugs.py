"""Regression tests for bugs found by the ChatGPT first-pass audit
(a_agent_ontology.prompt / replies_a_agent_ontology.txt).

No live LLM calls: `llm_client.embed` / `invoke_reasoning_agent` are
monkeypatched so these tests exercise only the generated agent's own control
flow, never a real network/provider call.

Bug 1+2 (call_embedding in agents/kb_react_agent.py and
agents_lg/minimal_solver_agent_lg.py): `call_embedding` reads
`_LLM_CLIENT_INFO_PRINTED` and later assigns to it without declaring it
`global`, so Python treats it as local for the whole function body and the
read raises UnboundLocalError -- silently caught by the outer `except`,
always returning `[[] for _ in texts]` instead of real embeddings. Root
cause: the `call_embedding` template in agent_ontology/instantiate.py is
missing `global _LLM_CLIENT_INFO_PRINTED` (present in the `call_llm`
template it was copied from).

Bug 3 (agents/kb_react_agent.py process_think_or_act / process_check_action):
`parse_response` only returns the JSON keys the LLM's response actually
included. `state.data.update(reasoning_agent_result)` therefore leaves any
schema field the LLM omitted (e.g. `tool_name` on a thought-only step, which
LLMs commonly omit despite instructions) at whatever value a PRIOR step left
in `state.data`. `process_check_action`'s gate reads that stale value and
wrongly re-executes the previous action instead of routing to record_step.
Root cause: the generic "Invoke agents (schema-aware)" merge block in
instantiate.py (both the custom and LangGraph backends) never resets a
schema's declared fields before merging.
"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agents"))
sys.path.insert(0, str(REPO_ROOT / "agents_lg"))


def _fresh_module(modname):
    """Import (or re-import) a generated agent module fresh, so monkeypatched
    module-level state (_LLM_CLIENT_INFO_PRINTED) starts from its real
    module-load default in every test."""
    if modname in sys.modules:
        del sys.modules[modname]
    return importlib.import_module(modname)


class TestCallEmbeddingGlobalDeclaration:
    """`call_embedding` must not crash with UnboundLocalError merely from
    reading its own module-level 'have we printed the banner yet' flag."""

    def test_kb_react_agent_call_embedding_returns_real_embeddings(self, monkeypatch):
        import llm_client

        fake_result = SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])
        monkeypatch.setattr(llm_client, "embed", lambda *a, **k: fake_result)

        m = _fresh_module("kb_react_agent")
        assert m._LLM_CLIENT_INFO_PRINTED is False

        out = m.call_embedding(["hello world"])

        assert out == [[0.1, 0.2, 0.3]], (
            f"expected the mocked embedding to come through, got {out!r} -- "
            "call_embedding is swallowing an UnboundLocalError instead of "
            "calling llm_client.embed"
        )

    def test_minimal_solver_agent_lg_call_embedding_returns_real_embeddings(self, monkeypatch):
        import llm_client

        fake_result = SimpleNamespace(embeddings=[[0.4, 0.5]])
        monkeypatch.setattr(llm_client, "embed", lambda *a, **k: fake_result)

        m = _fresh_module("minimal_solver_agent_lg")
        assert m._LLM_CLIENT_INFO_PRINTED is False

        out = m.call_embedding(["another text"])

        assert out == [[0.4, 0.5]], (
            f"expected the mocked embedding to come through, got {out!r} -- "
            "call_embedding is swallowing an UnboundLocalError instead of "
            "calling llm_client.embed"
        )


class TestReActStepDoesNotLeakStaleFields:
    """A thought-only step (LLM response omits tool_name/tool_input, as real
    LLMs commonly do despite the system prompt's instructions) must not be
    routed as an action re-executing the PREVIOUS step's tool call."""

    def test_thought_after_action_does_not_replay_stale_tool_name(self, monkeypatch):
        m = _fresh_module("kb_react_agent")

        state = m.AgentState()
        state.data["query"] = "irrelevant for this test"

        # Simulate the tail end of a prior action step: tool_name/tool_input
        # are populated exactly as process_think_or_act would leave them.
        state.data["type"] = "action"
        state.data["tool_name"] = "kb_lookup"
        state.data["tool_input"] = "comp_nexus"
        state.data["action"] = "kb lookup comp_nexus"

        # The LLM's *next* response is a thought-only step. Real providers
        # frequently omit fields they consider empty/irrelevant rather than
        # emitting `"tool_name": ""` explicitly.
        monkeypatch.setattr(
            m,
            "invoke_reasoning_agent",
            lambda user_message, output_schema=None: (
                '{"type": "thought", "thought": "I have enough info now."}'
            ),
        )

        state = m.process_think_or_act(state)
        branch = m.process_check_action(state)

        assert branch == "record_step", (
            f"got branch={branch!r}; a thought-only step (LLM omitted "
            f"tool_name) was routed to re-execute the STALE prior action "
            f"instead of record_step. state.data['tool_name']="
            f"{state.data.get('tool_name')!r}"
        )
