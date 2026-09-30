"""Regression tests for round-2 ChatGPT audit findings
(chatgpt-bug-audit/replies_ao2.txt).

No live LLM calls: `invoke_solver` / `invoke_reasoning_agent` are
monkeypatched so these tests exercise only the generated agent's own control
flow, never a real network/provider call.

Finding 1 (agents/kb_react_agent.py process_emit_answer, and sibling sites
agents/react_agent.py, agents_lg/kb_react_agent_lg.py,
agents_lg/react_agent_lg.py): the emit-answer step does
`state.data.get("answer", state.data.get("thought", "No answer found"))`.
`.get(key, default)` only falls back when the key is *missing*, not when it
is present but falsy/empty. `process_think_or_act` resets
`state.data["answer"] = ""` on every step (custom backend) /
`updates["answer"] = ""` (langgraph backend), so if the agent reaches
max_steps without ever producing an "answer" step, the key IS present (as
"") and the fallback never triggers.

The computed fallback is also never written back to `state.data["answer"]`
(custom backend) / `updates["answer"]` (langgraph backend) -- only printed.
That matters because downstream consumers read `state.data["answer"]`
directly: `agent_ontology/benchmarks/scoring.py` scores the run against it,
and `tests/test_agents.py`'s per-agent health checks (e.g. lines ~213, 377,
384) treat a missing/empty `state.data["answer"]` as a failed run. So this
step is where "the run terminates with an empty final answer" actually
becomes true for every downstream consumer, not merely for the console
printout -- the fix persists the fallback into state, not just the local
variable used for printing.

Root cause: this logic is authored verbatim in the agent spec YAML
(`agent_ontology/specs/kb_react.yaml` and `agent_ontology/specs/react.yaml`
`emit_answer.logic`), which instantiate.py copies near-verbatim (custom
backend) or transforms mechanically (langgraph backend via
`_transform_logic_line_lg`) into the generated files -- instantiate.py's own
Python-emitting code contains no template string for this fallback. Fixed at
the spec source (the true template for this business logic) and regenerated
all four affected generated files.

Finding 2 (agents_lg/minimal_solver_agent_lg.py run(), and all 37
agents_lg/*.py sibling files sharing the same generic instantiate.py
`generate_langgraph_agent` run() template): `run()` wraps `app.invoke(...)`
in `try/except Exception`, substitutes the untouched initial state on
failure, then unconditionally sets `_clean_exit = True` -- a real exception
(e.g. a node raising TypeError) is swallowed and reported as a clean exit
with no error indication.

Root cause: instantiate.py's `generate_langgraph_agent` run() template
(unlike the custom-backend run() template, which correctly derives
`_clean_exit` from whether MAX_ITERATIONS was hit and lets node exceptions
propagate uncaught). Fixed at the generator template and regenerated all 37
agents_lg/*.py files that share it.
"""
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agents"))
sys.path.insert(0, str(REPO_ROOT / "agents_lg"))


def _fresh_module(modname):
    if modname in sys.modules:
        del sys.modules[modname]
    return importlib.import_module(modname)


class TestEmitAnswerFallsBackToThoughtNotEmptyString:
    """When max_steps is reached without ever producing a real "answer" step,
    the emit-answer logic must fall back to the current thought (or an
    explicit failure indicator), never silently return the empty string left
    behind by the previous think_or_act step."""

    def test_kb_react_agent_custom_backend(self, monkeypatch):
        m = _fresh_module("kb_react_agent")
        state = m.AgentState()
        # Simulate: think_or_act already reset answer to "" (as it does on
        # every step), and the agent's last thought was never promoted to a
        # real answer because max_steps was hit on a thought/action step.
        state.data["answer"] = ""
        state.data["thought"] = "I believe comp_nexus was founded in 2019."

        m.process_emit_answer(state)

        assert state.data["answer"] != "", (
            "process_emit_answer returned the empty string left behind by "
            "think_or_act instead of falling back to the current thought "
            "-- .get(key, default) only falls back on a MISSING key, not a "
            "present-but-empty one"
        )
        assert "2019" in state.data["answer"] or "comp_nexus" in state.data["answer"], (
            f"expected the fallback answer to carry the thought's content, "
            f"got {state.data['answer']!r}"
        )

    def test_react_agent_custom_backend(self, monkeypatch):
        m = _fresh_module("react_agent")
        state = m.AgentState()
        state.data["answer"] = ""
        state.data["thought"] = "My best guess is 42."

        m.process_emit_answer(state)

        assert state.data["answer"] != "", (
            "react_agent's process_emit_answer has the same "
            ".get(key, default) truthiness-fallback bug as kb_react_agent"
        )

    def test_kb_react_agent_lg_backend(self):
        m = _fresh_module("kb_react_agent_lg")
        state = {"answer": "", "thought": "Best guess: comp_nexus, 2019."}

        result = m.node_emit_answer(state)

        assert result.get("answer", "") != "", (
            "kb_react_agent_lg's emit-answer node has the same "
            "truthiness-fallback bug as the custom backend, and/or never "
            "writes the fallback back into the returned updates dict"
        )

    def test_react_agent_lg_backend(self):
        m = _fresh_module("react_agent_lg")
        state = {"answer": "", "thought": "Best guess: 42."}

        result = m.node_emit_answer(state)

        assert result.get("answer", "") != "", (
            "react_agent_lg's emit-answer node has the same "
            "truthiness-fallback bug as the custom backend, and/or never "
            "writes the fallback back into the returned updates dict"
        )


class TestLangGraphRunDoesNotSwallowExceptions:
    """A real exception escaping a graph node must not be reported as a
    clean exit with no trace of failure."""

    def test_minimal_solver_agent_lg_propagates_or_marks_unclean(self, monkeypatch):
        m = _fresh_module("minimal_solver_agent_lg")

        def _boom(user_message, output_schema=None):
            raise TypeError("simulated node failure (e.g. 123[:100])")

        monkeypatch.setattr(m, "invoke_solver", _boom)

        m.TRACE.clear()
        captured_clean_exit = {}
        orig_dump_trace = m.dump_trace

        def _spy_dump_trace(path="trace.json", iterations=0, clean_exit=True):
            captured_clean_exit["value"] = clean_exit
            return orig_dump_trace(path=path, iterations=iterations, clean_exit=clean_exit)

        monkeypatch.setattr(m, "dump_trace", _spy_dump_trace)

        try:
            result = m.run({"problem": "2+2"})
        except TypeError:
            # Propagating the exception is an acceptable fix too.
            return

        assert captured_clean_exit.get("value") is False, (
            f"a TypeError raised inside a graph node was swallowed and "
            f"reported as clean_exit={captured_clean_exit.get('value')!r} "
            f"-- the failure must propagate or be marked unclean, not "
            f"silently reported as a successful run"
        )
