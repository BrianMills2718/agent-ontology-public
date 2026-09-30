"""Runtime contract every generated agent must keep, checked over ALL of them.

Three ChatGPT audit rounds found defects in the same place -- the runtime code
`agent_ontology/instantiate.py` emits into every agent -- and each fix landed in
one or two generated files while the other ~70 kept the bug:

- round 1: `call_embedding` raised UnboundLocalError that was silently caught;
  a thought-only ReAct step re-used the previous step's `tool_name`.
- round 2: an empty final answer at max_steps; LangGraph `run()` swallowed a
  node exception and reported a clean exit.
- round 3: a thought-only ReAct step recorded the previous step's tool
  `observation`.

So this file checks the contract against every generated agent in both
backends rather than against the one file an audit happened to quote:

1. A failure inside the agent is not converted into a normal return. The LLM
   provider raising is the most common real failure; it must escape `run()`,
   which means `python agents/<name>.py` exits non-zero instead of printing a
   plausible-looking "answer" built from an error stub.
2. Tool-exchange state (`observation`, `tool_results`, `tool_calls`) belongs to
   one agent decision. A later decision that runs no tool must not inherit it.
3. The "reset omitted output fields" rule from round 1 must not wipe fields the
   agent was only handed as context (its declared input) -- otherwise a
   verifier that omits `answer` erases the answer it was asked to check.

No live LLM calls: `llm_client.call_llm` is replaced for the duration of each
test. Every generated agent imports it lazily inside `call_llm`, so replacing
the module attribute reaches all of them.
"""

from __future__ import annotations

import builtins
import copy
import importlib
import importlib.util
import json
import runpy
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

llm_client = pytest.importorskip("llm_client")

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_FILES = sorted((REPO_ROOT / "agents").glob("*_agent.py")) + sorted(
    (REPO_ROOT / "agents_lg").glob("*_agent_lg.py")
)


class _ProviderDown(RuntimeError):
    """Stands in for any provider failure: auth, quota, network, budget."""


def _ids(paths):
    return [f"{p.parent.name}/{p.name}" for p in paths]


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Run an agent with its trace/workspace files kept out of the repo."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ONTOLOGY_MOCK_MODE", "deterministic")
    monkeypatch.setattr(builtins, "input", _fake_input)
    for key in [k for k in sys.modules if k.endswith(("_agent", "_agent_lg"))]:
        monkeypatch.delitem(sys.modules, key)
    return tmp_path


class _Alarm(Exception):
    pass


def _with_deadline(seconds, fn):
    def _handler(signum, frame):
        raise _Alarm(f"agent did not finish within {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ── 1. Failures propagate ────────────────────────────────────────────────


def _blocked_agent(agent_path):
    """Unimplemented processes on this generated agent's path (section 7)."""
    import yaml

    sys.path.insert(0, str(REPO_ROOT))
    from agent_ontology.instantiate import unimplemented_reachable_processes

    stem = agent_path.stem.removesuffix("_lg").removesuffix("_agent")
    for name in (stem, stem.replace("_", "-")):
        spec_path = REPO_ROOT / "agent_ontology" / "specs" / f"{name}.yaml"
        if spec_path.exists():
            return unimplemented_reachable_processes(yaml.safe_load(spec_path.read_text()))
    return []

_QUESTION = "What is 2 + 2?"
_SEED = {
    "query": _QUESTION, "question": _QUESTION, "problem": _QUESTION, "objective": _QUESTION,
    "max_tasks": 1,
    "tasks": [{"id": 1, "description": _QUESTION, "status": "pending", "priority": 1}],
    # Self Improver's input is a spec to improve.
    "spec_yaml": (REPO_ROOT / "agent_ontology" / "specs" / "minimal_solver.yaml").read_text(),
}


def _fake_input(prompt=""):
    """`__main__` asks `input("Enter <field>: ")` per input field."""
    field = str(prompt).removeprefix("Enter ").rstrip(": ").strip()
    return _SEED.get(field, _QUESTION)


@pytest.mark.parametrize("agent_path", AGENT_FILES, ids=_ids(AGENT_FILES))
def test_llm_failure_is_not_reported_as_a_normal_run(agent_path, isolated, monkeypatch):
    """Run the agent as `python <file>` does. The provider raises on every call.

    Before this contract, `call_llm` caught every exception and returned
    `{"stub": true, "error": ...}` as though the model had answered. The agent
    parsed that, walked its graph with empty fields, emitted a fallback answer
    and exited 0; the benchmark then scored an outage as a wrong answer. The
    LangGraph `run()` additionally caught any node exception and returned the
    untouched input state.
    """
    calls = {"n": 0}

    def _down(*args, **kwargs):
        calls["n"] += 1
        raise _ProviderDown("simulated provider outage")

    monkeypatch.setattr(llm_client, "call_llm", _down, raising=False)

    try:
        try:
            _with_deadline(60, lambda: runpy.run_path(str(agent_path), run_name="__main__"))
        except (_ProviderDown, SystemExit, _Alarm):
            raise
        except Exception:
            # A bare `__main__` may reject missing inputs before any LLM call
            # (Self Improver needs a spec). That is loud, not swallowed; fall
            # through to the seeded run, which must reach the provider.
            if calls["n"]:
                raise
        if calls["n"] == 0:
            # Some `__main__` blocks take no input (BabyAGI starts from a task
            # list), so a bare run has nothing to do. Seed the common inputs.
            spec = importlib.util.spec_from_file_location(f"_contract_{agent_path.stem}", agent_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _with_deadline(60, lambda: module.run(copy.deepcopy(_SEED)))
    except _ProviderDown:
        return  # the contract: the failure escaped, so the process exits non-zero
    except NotImplementedError:
        # Also loud: the agent reaches a process the generator does not
        # implement before any LLM call (see section 7). Only allowed for
        # agents the spec says are blocked that way.
        if not _blocked_agent(agent_path):
            raise
        return
    except SystemExit as exc:
        assert exc.code not in (0, None), f"{agent_path.name} exited 0 after a provider failure"
        return

    if calls["n"] == 0:
        pytest.fail(
            f"{agent_path.name} finished without ever calling the LLM, even with seeded inputs; "
            "this test cannot check its failure path -- give it an input that reaches an LLM call"
        )
    pytest.fail(
        f"{agent_path.name} returned normally after {calls['n']} failed LLM call(s): "
        "the provider failure was swallowed and the run looked successful"
    )


# ── 2. Tool-exchange state is per decision ───────────────────────────────


def _scripted(monkeypatch, replies):
    """Replace the provider with a fixed sequence of JSON replies."""
    queue = [json.dumps(r) for r in replies]

    def _fake(model, messages, **kwargs):
        if not queue:
            raise AssertionError("agent asked for more LLM turns than the script provides")
        return SimpleNamespace(content=queue.pop(0))

    monkeypatch.setattr(llm_client, "call_llm", _fake, raising=False)


REACT_FAMILY = [
    ("agents", "react_agent"),
    ("agents", "kb_react_agent"),
    ("agents_lg", "react_agent_lg"),
    ("agents_lg", "kb_react_agent_lg"),
]


def _load(folder, modname, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / folder))
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    return importlib.import_module(modname)


@pytest.mark.parametrize("folder,modname", REACT_FAMILY)
def test_thought_step_after_action_has_no_observation(folder, modname, isolated, monkeypatch):
    """Round-3 finding: action -> thought recorded the action's observation twice."""
    m = _load(folder, modname, monkeypatch)
    _scripted(monkeypatch, [
        {"type": "action", "thought": "compute", "action": "calculate 1+1",
         "tool_name": "calculate", "tool_input": "1+1"},
        {"type": "thought", "thought": "that gave me a number"},
        {"type": "answer", "answer": "2"},
    ])

    state = m.run({"query": "What is 1+1?", "question": "What is 1+1?"})

    trajectory = state.data["trajectory"]
    assert len(trajectory) == 2, trajectory
    assert trajectory[0]["observation"], "the action step should carry its tool output"
    assert trajectory[1]["observation"] == "", (
        f"thought-only step inherited the previous action's observation: {trajectory[1]!r}"
    )


@pytest.mark.parametrize("folder,modname", REACT_FAMILY)
def test_tool_calls_batch_is_not_replayed_on_the_next_action(folder, modname, isolated, monkeypatch):
    """`tool_calls` is read by the dispatcher before `tool_name`. A batch from
    one decision must not be dispatched again for the next, different action."""
    m = _load(folder, modname, monkeypatch)
    _scripted(monkeypatch, [
        {"type": "action", "thought": "batch",
         "tool_calls": [{"tool_name": "calculate", "arguments": "1+1"}]},
        {"type": "action", "thought": "single", "tool_name": "calculate", "tool_input": "20+22"},
        {"type": "answer", "answer": "42"},
    ])

    state = m.run({"query": "q", "question": "q"})

    trajectory = state.data["trajectory"]
    assert len(trajectory) == 2, trajectory
    assert "42" in str(trajectory[1]["observation"]), (
        f"second action re-ran the first decision's tool_calls batch: {trajectory[1]!r}"
    )


@pytest.mark.parametrize("folder,modname", REACT_FAMILY)
def test_action_step_records_which_tool_ran(folder, modname, isolated, monkeypatch):
    """The prompt tells the model to set tool_name/tool_input for an action, not
    the free-text `action` field -- so record_step, which saved only `action`,
    fed the model observations with no record of what produced them."""
    m = _load(folder, modname, monkeypatch)
    _scripted(monkeypatch, [
        {"type": "action", "thought": "multiply", "tool_name": "calculate", "tool_input": "6*7"},
        {"type": "answer", "answer": "42"},
    ])

    state = m.run({"query": "q", "question": "q"})

    step = state.data["trajectory"][0]
    assert "calculate" in step["action"] and "6*7" in step["action"], step


@pytest.mark.parametrize("folder,modname", REACT_FAMILY)
def test_max_steps_without_answer_is_not_empty(folder, modname, isolated, monkeypatch):
    """Round-2 finding, now checked through run() in both backends."""
    m = _load(folder, modname, monkeypatch)
    _scripted(monkeypatch, [{"type": "thought", "thought": f"still thinking {i}"} for i in range(10)])

    state = m.run({"query": "q", "question": "q"})

    assert state.data.get("answer"), f"run ended at max_steps with an empty answer: {state.data.get('answer')!r}"


# ── 3. Resetting omitted outputs must not erase declared inputs ──────────


@pytest.mark.parametrize("folder,modname", [
    ("agents", "multidoc_structured_agent"),
    ("agents_lg", "multidoc_structured_agent_lg"),
])
def test_verifier_that_omits_answer_does_not_erase_it(folder, modname, isolated, monkeypatch):
    """`verify` receives ReasoningOutput{answer, reasoning} and returns the same
    schema. A verifier reply that omits `answer` means "unchanged", not "empty";
    blanket-resetting every output field turned it into an empty final answer."""
    m = _load(folder, modname, monkeypatch)
    node = getattr(m, "process_verify", None) or getattr(m, "node_verify")
    _scripted(monkeypatch, [{"reasoning": "checked against the cards; consistent"}])

    if hasattr(m, "process_verify"):
        state = m.AgentState()
        state.data.update({"answer": "Paris", "reasoning": "first pass", "question": "q"})
        node(state)
        answer = state.data.get("answer")
    else:
        updates = node({"answer": "Paris", "reasoning": "first pass", "question": "q"})
        answer = updates.get("answer", "Paris")

    assert answer == "Paris", f"verifier omission erased the answer under review: {answer!r}"


# ── 4. LangGraph nodes run the spec logic the custom backend runs ────────
#
# These were found by test 1 above: once `run()` stopped swallowing
# exceptions, three LangGraph agents turned out never to have worked. Each is a
# rewrite rule in instantiate.py's `_transform_logic_line_lg`, so each test pins
# the rule through one real generated agent.


def test_lg_node_reads_its_own_earlier_writes(isolated, monkeypatch):
    """`current = state.data["original"]` right after `state.data["original"] = ...`
    read the pre-node state in LangGraph, so Self Improver analyzed an empty spec."""
    m = _load("agents_lg", "self_improver_agent_lg", monkeypatch)
    updates = m.node_receive_spec({"spec_yaml": "name: x\n"})
    assert updates["current_spec_yaml"] == "name: x\n", updates


def test_lg_nested_write_goes_to_updates(isolated, monkeypatch):
    """`state.data["pr_metadata"]["diff"] = ...` became
    `state.get('pr_metadata', '')["diff"] = ...`: TypeError on a str, so Code
    Reviewer (LangGraph) crashed on its first node for every input."""
    m = _load("agents_lg", "code_reviewer_agent_lg", monkeypatch)
    updates = m.node_intake_pr({"diff": "d", "files": ["a.py"], "description": "why"})
    assert updates["pr_metadata"] == {"diff": "d", "files": ["a.py"], "description": "why"}




def test_lg_graph_with_reserved_field_name_compiles(isolated, monkeypatch):
    """A spec field named `checkpoint_id` became a state channel LangGraph
    reserves, so SOTA Superset (LangGraph) could not even build its graph."""
    m = _load("agents_lg", "sota_superset_runtime_agent_lg", monkeypatch)
    m.build_graph()


# ── 5. What a step persists is the record it keeps ───────────────────────
#
# Round 4: kb_react's record_step appended a local `step` dict to the in-memory
# trajectory, but the generated store write rebuilt the persisted entry from
# state.data by field name. The two disagreed whenever the local record was
# computed (the synthesized "tool[input]" action) or renamed (debate's
# "side": "con", reflexion's trial -> trial_number, mixture's current_layer ->
# layer). The contract: when a step builds a record shaped like the schema it
# persists, the store receives that record.


def _persisting_record_steps():
    """(spec, step, store, schema) for every step whose logic assigns a dict
    literal covering all fields of a schema it writes to a store. Found by
    parsing the spec, independently of the generator."""
    import ast
    import yaml

    found = []
    for spec_path in sorted((REPO_ROOT / "agent_ontology" / "specs").glob("*.yaml")):
        spec = yaml.safe_load(spec_path.read_text())
        schemas = {s["name"]: {f["name"] for f in s.get("fields", [])} for s in spec.get("schemas", [])}
        procs = {p["id"]: p for p in spec.get("processes", [])}
        for edge in spec.get("edges", []):
            if edge.get("type") != "write" or edge.get("data") not in schemas:
                continue
            logic = procs.get(edge["from"], {}).get("logic") or ""
            import textwrap
            body = ast.parse("def _f():\n" + textwrap.indent(logic or "pass", "    ")).body[0].body
            for node in body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                    keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
                    if schemas[edge["data"]] <= keys:
                        found.append((spec_path.stem, edge["from"], edge["to"], edge["data"]))
                        break
    return found


PERSISTING_STEPS = _persisting_record_steps()


def test_persisting_record_steps_were_found():
    """Guard the guard: the scan must keep seeing the known cases."""
    names = {(s, p) for s, p, _, _ in PERSISTING_STEPS}
    assert {("kb_react", "record_step"), ("react", "record_step"), ("debate", "record_con_argument"),
            ("reflexion", "store_reflection"), ("mixture_of_agents", "record_layer")} <= names, names


def _new_dict_entries(before, after):
    """Dicts appended to any list in state by the step."""
    out = []
    for key, val in after.items():
        if isinstance(val, list) and val and isinstance(val[-1], dict):
            if len(val) > len(before.get(key) or []):
                out.append(val[-1])
    return out


@pytest.mark.parametrize("backend", ["custom", "langgraph"])
@pytest.mark.parametrize("spec_name,step,store,schema", PERSISTING_STEPS,
                         ids=[f"{s}.{p}->{st}" for s, p, st, _ in PERSISTING_STEPS])
def test_persisted_entry_matches_the_record_the_step_keeps(spec_name, step, store, schema, backend,
                                                           isolated, monkeypatch):
    folder, suffix = ("agents", "_agent") if backend == "custom" else ("agents_lg", "_agent_lg")
    m = _load(folder, f"{spec_name.replace('-', '_')}{suffix}", monkeypatch)
    fields = [f["name"] for f in m.SCHEMAS[schema]["fields"]]
    # Top-level values that a correct write must NOT pick up in place of the record's.
    seed = {f: f"STALE_{f}" for f in fields}
    seed.update({"tool_name": "calculate", "tool_input": "6*7", "action": "", "trajectory": [],
                 "debate_history": [], "episodes": [], "layer_history": [], "reflections": []})

    written = []
    if backend == "custom":
        state = m.AgentState()
        state.data.update(seed)
        monkeypatch.setattr(getattr(state, store), "write", lambda v, key=None: written.append(v))
        before = {k: list(v) if isinstance(v, list) else v for k, v in state.data.items()}
        getattr(m, f"process_{step}")(state)
        after = state.data
    else:
        monkeypatch.setattr(getattr(m, f"_store_{store}"), "write", lambda v, key=None: written.append(v))
        before = copy.deepcopy(seed)
        after = {**seed, **getattr(m, f"node_{step}")(copy.deepcopy(seed))}

    assert len(written) == 1, written
    # The kept record is either appended to a list in state, or assigned to a
    # state key as a whole dict (multi_agent_codegen's final_package).
    kept = [e for e in _new_dict_entries(before, after) if set(fields) <= set(e)]
    kept += [v for k, v in after.items()
             if isinstance(v, dict) and set(fields) <= set(v) and k not in before]
    assert kept, f"{step} persisted a {schema} but appended no matching record to state"
    assert written[0] == {f: kept[-1][f] for f in fields}, (
        f"persisted {schema} disagrees with the record kept in state:\n"
        f"  persisted: {written[0]}\n  kept:      {kept[-1]}"
    )


@pytest.mark.parametrize("folder,modname", REACT_FAMILY)
def test_persisted_trajectory_matches_in_memory_trajectory(folder, modname, isolated, monkeypatch):
    """End to end through run(): every persisted TrajectoryEntry equals the
    corresponding in-memory trajectory step."""
    m = _load(folder, modname, monkeypatch)
    _scripted(monkeypatch, [
        {"type": "action", "thought": "multiply", "tool_name": "calculate", "tool_input": "6*7"},
        {"type": "thought", "thought": "got it"},
        {"type": "answer", "answer": "42"},
    ])

    state = m.run({"query": "q", "question": "q"})

    store = state.trajectory_store if hasattr(state, "trajectory_store") else m._store_trajectory_store
    assert store.queue == state.data["trajectory"], (store.queue, state.data["trajectory"])


# ── 6. A schema-invalid LLM reply is never merged or routed ──────────────
#
# Round 5: `validate_output` counted violations and the run carried on, so
# {"type": "answer", "answer": 42} was merged and emit_answer crashed on
# `answer[:200]` -- or, with a wrong enum, the step was routed on a value its
# gate never expected. The contract: a reply whose present fields violate
# their declared types raises SchemaViolationError naming the schema and field
# before anything is merged. (An omitted field or a null is not a violation:
# omitted outputs reset to their defaults, per section 3.)


def _bad_value(ftype, schema_names):
    """A JSON value that violates `ftype`, or None if the type is unchecked."""
    if ftype in ("string", "datetime"):
        return 12345
    if ftype in ("integer", "float", "number"):
        return "not-a-number"
    if ftype == "boolean":
        return "yes"
    if ftype.startswith("list") or ftype == "array":
        return "not-a-list"
    if ftype.startswith("enum["):
        return "__not_an_allowed_value__"
    if ftype == "object" or ftype in schema_names:
        return "not-an-object"
    return None


def _good_value(ftype, schema_names):
    if ftype.startswith("enum["):
        return ftype[5:-1].split(",")[0].strip()
    return {"string": "ok", "datetime": "2026-01-01", "integer": 1, "float": 1.0, "number": 1,
            "boolean": True, "array": [], "object": {}}.get(
        ftype, [] if ftype.startswith("list") else ({} if ftype in schema_names else "ok"))


def _llm_output_cases():
    """(spec, module stem, process, agent, schema, field, bad value) for every
    agent invoke with a declared output schema, violating its first checkable field."""
    import yaml

    cases = []
    for spec_path in sorted((REPO_ROOT / "agent_ontology" / "specs").glob("*.yaml")):
        spec = yaml.safe_load(spec_path.read_text())
        schemas = {s["name"]: s for s in spec.get("schemas", [])}
        entities = {e["id"]: e for e in spec.get("entities", [])}
        # Only `step` processes call an LLM in generated code; `spawn` processes
        # are emitted as stubs that call nothing.
        steps = {p["id"] for p in spec.get("processes", []) if p.get("type") == "step"}
        seen = set()
        for edge in spec.get("edges", []):
            if edge.get("type") != "invoke" or entities.get(edge["to"], {}).get("type") != "agent":
                continue
            if edge["from"] not in steps:
                continue
            out = edge.get("output")
            if out not in schemas or (edge["from"], out) in seen:
                continue
            seen.add((edge["from"], out))
            for f in schemas[out].get("fields", []):
                bad = _bad_value(f["type"], schemas)
                if bad is not None:
                    cases.append((spec_path.stem.replace("-", "_"), edge["from"], edge["to"], out, f["name"], bad))
                    break
    return cases


LLM_OUTPUT_CASES = _llm_output_cases()


def test_llm_output_cases_cover_the_known_finding():
    assert len(LLM_OUTPUT_CASES) > 40, len(LLM_OUTPUT_CASES)
    assert any(c[0] == "kb_react" and c[3] == "ReActStep" for c in LLM_OUTPUT_CASES)


@pytest.mark.parametrize("backend", ["custom", "langgraph"])
@pytest.mark.parametrize("spec_name,process,agent,schema,field,bad", LLM_OUTPUT_CASES,
                         ids=[f"{c[0]}.{c[1]}:{c[3]}.{c[4]}" for c in LLM_OUTPUT_CASES])
def test_type_violating_reply_is_loud_and_not_merged(spec_name, process, agent, schema, field, bad,
                                                     backend, isolated, monkeypatch):
    folder, suffix = ("agents", "_agent") if backend == "custom" else ("agents_lg", "_agent_lg")
    m = _load(folder, f"{spec_name}{suffix}", monkeypatch)
    schema_names = set(m.SCHEMAS)

    def reply_for(schema_name):
        fields = m.SCHEMAS[schema_name]["fields"]
        reply = {f["name"]: _good_value(f["type"], schema_names) for f in fields}
        if schema_name == schema:
            reply[field] = bad
        return json.dumps(reply)

    # Every agent wrapper in the module answers by the schema it is asked for;
    # only the target schema's reply is invalid.
    for name in [n for n in dir(m) if n.startswith("invoke_")]:
        monkeypatch.setattr(m, name, lambda msg, output_schema=None: reply_for(output_schema or schema))

    # Seed every declared field with a valid value so the step's own logic
    # reaches its LLM call.
    seed = {}
    for s in m.SCHEMAS.values():
        for f in s["fields"]:
            seed.setdefault(f["name"], _good_value(f["type"], schema_names))
    seed.update(copy.deepcopy(_SEED))
    marker = f"{process}_result"

    if backend == "custom":
        state = m.AgentState()
        state.data.update(seed)
        call = lambda: getattr(m, f"process_{process}")(state)  # noqa: E731
    else:
        call = lambda: getattr(m, f"node_{process}")(dict(seed))  # noqa: E731

    with pytest.raises(m.SchemaViolationError) as err:
        _with_deadline(30, call)

    assert schema in str(err.value) and field in str(err.value), str(err.value)
    if backend == "custom":
        assert state.data.get(field) != bad, f"invalid {schema}.{field}={bad!r} was merged into state"
        assert state.data.get(marker, {}).get(field) != bad if isinstance(state.data.get(marker), dict) else True


def test_generator_refuses_a_step_whose_agent_call_is_dead(monkeypatch):
    """Plan-and-Solve's synthesize step set `_done = True` before its agent call;
    the generated step returned first, so the synthesizer never ran."""
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    from agent_ontology.instantiate import _check_invoke_reachable

    with pytest.raises(ValueError, match="can never run"):
        _check_invoke_reachable({"id": "s", "logic": 'x = 1\nstate.data["_done"] = True\n'}, True)
    _check_invoke_reachable({"id": "s", "logic": 'if not x:\n    state.data["_done"] = True\n'}, True)


# ── 7. No placeholder step reports progress it did not make ──────────────
#
# Spawn processes were generated as `print("[SPAWN] Would create sub-agent")`
# followed by a normal return; formal_translation, symbolic_inference, search,
# protocol and error_handler became empty functions. map_reduce never mapped,
# pddl_planner never planned, alpha_geometry never deduced -- and each run
# reported a clean exit. The contract: a process type the generator cannot
# implement raises NotImplementedError naming the agent and the process, and
# no agent that reaches one is listed as runnable by the benchmark.

_IMPLEMENTED_TYPES = {"step", "gate", "checkpoint", "policy"}


def _unimplemented_cases():
    import yaml

    cases = []
    for spec_path in sorted((REPO_ROOT / "agent_ontology" / "specs").glob("*.yaml")):
        spec = yaml.safe_load(spec_path.read_text())
        for p in spec.get("processes", []):
            if p.get("type") not in _IMPLEMENTED_TYPES:
                cases.append((spec_path.stem.replace("-", "_"), p["id"], p["type"], spec.get("name", "")))
    return cases


UNIMPLEMENTED_CASES = _unimplemented_cases()


def test_unimplemented_cases_include_the_spawns():
    ids = {(c[0], c[1]) for c in UNIMPLEMENTED_CASES}
    assert {("map_reduce", "map_chunks"), ("meta_prompting", "recurse_decomposition"),
            ("claude_code", "spawn_sub_agent"), ("pddl_planner", "run_planner")} <= ids, ids


@pytest.mark.parametrize("backend", ["custom", "langgraph"])
@pytest.mark.parametrize("spec_name,process,ptype,agent_name", UNIMPLEMENTED_CASES,
                         ids=[f"{c[0]}.{c[1]}:{c[2]}" for c in UNIMPLEMENTED_CASES])
def test_unimplemented_process_fails_loudly(spec_name, process, ptype, agent_name, backend,
                                            isolated, monkeypatch):
    folder, suffix = ("agents", "_agent") if backend == "custom" else ("agents_lg", "_agent_lg")
    m = _load(folder, f"{spec_name}{suffix}", monkeypatch)
    if backend == "custom":
        state = m.AgentState()
        state.data.update(copy.deepcopy(_SEED))
        call = lambda: getattr(m, f"process_{process}")(state)  # noqa: E731
    else:
        call = lambda: getattr(m, f"node_{process}")(copy.deepcopy(_SEED))  # noqa: E731
    with pytest.raises(NotImplementedError) as err:
        call()
    assert process in str(err.value) and agent_name in str(err.value), str(err.value)


def _step_ids_by_agent_file():
    """Generated file stem -> ids of `step` processes. A step's body is its
    spec's own logic, so a step that only logs its input is the author's intent."""
    import yaml

    out = {}
    for spec_path in (REPO_ROOT / "agent_ontology" / "specs").glob("*.yaml"):
        spec = yaml.safe_load(spec_path.read_text())
        steps = {p["id"] for p in spec.get("processes", []) if p.get("type") == "step"}
        stem = spec_path.stem.replace("-", "_")
        out[f"{stem}_agent"] = out[f"{stem}_agent_lg"] = steps
    return out


_STEP_IDS = _step_ids_by_agent_file()


def _placeholder_functions(path):
    """Non-step process_*/node_* functions whose body does nothing but
    print/pass/return, and any printed "would ..." placeholder."""
    import ast

    steps = _STEP_IDS.get(path.stem, set())
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef) and node.name.startswith(("process_", "node_"))
                and node.name.split("_", 1)[1] not in steps):
            body = [s for s in node.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            inert = all(
                isinstance(s, (ast.Pass, ast.Return))
                or (isinstance(s, ast.Assign) and ast.unparse(s.value) == "{}")
                or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and ast.unparse(s.value.func) == "print")
                for s in body
            )
            if inert:
                found.append(f"{node.name}: does nothing")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            import re
            if re.search(r"\bwould (create|run|call|spawn|execute|invoke)\b", node.value, re.I):
                found.append(f"placeholder text: {node.value.strip()[:60]}")
    return found


@pytest.mark.parametrize("agent_path", AGENT_FILES, ids=_ids(AGENT_FILES))
def test_no_generated_step_is_a_silent_placeholder(agent_path):
    assert _placeholder_functions(agent_path) == []


def test_benchmark_lists_no_agent_that_cannot_run(monkeypatch):
    """An agent whose reachable path includes an unimplemented process cannot
    produce a meaningful score, so the benchmark must not list it."""
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    import yaml
    from agent_ontology.instantiate import unimplemented_reachable_processes
    from agent_ontology import benchmark
    from agent_ontology.benchmarks.compatibility import COMPATIBILITY

    not_runnable = {}
    for spec_path in sorted((REPO_ROOT / "agent_ontology" / "specs").glob("*.yaml")):
        blocked = unimplemented_reachable_processes(yaml.safe_load(spec_path.read_text()))
        if blocked:
            not_runnable[spec_path.stem.replace("-", "_")] = blocked
    assert "map_reduce" in not_runnable, not_runnable
    listed = set(benchmark.KNOWN_AGENTS) | set(COMPATIBILITY)
    assert not (listed & set(not_runnable)), {a: not_runnable[a] for a in listed & set(not_runnable)}


# ── 8. A reply that is not a JSON object of the declared schema is loud ──
#
# Round 6: parse_response turns a non-JSON reply into {"raw": text};
# require_valid_output exempted it, the reset rule filled every field with its
# default, and a ReAct run recorded an empty thought and ended "No answer
# found" -- a format failure scored as an ordinary wrong answer. The contract:
# for a declared output schema, a reply that is not a JSON object raises
# SchemaViolationError naming the agent and schema with the start of the text.
# One exception, where prose is unambiguous: a schema whose only field is a
# string takes the non-empty reply text as that field.


def _schema_output_cases():
    import yaml

    cases = []
    for spec_path in sorted((REPO_ROOT / "agent_ontology" / "specs").glob("*.yaml")):
        spec = yaml.safe_load(spec_path.read_text())
        schemas = {s["name"]: s for s in spec.get("schemas", [])}
        entities = {e["id"]: e for e in spec.get("entities", [])}
        steps = {p["id"] for p in spec.get("processes", []) if p.get("type") == "step"}
        seen = set()
        for edge in spec.get("edges", []):
            if (edge.get("type") == "invoke" and entities.get(edge["to"], {}).get("type") == "agent"
                    and edge["from"] in steps and edge.get("output") in schemas
                    and (edge["from"], edge["output"]) not in seen):
                seen.add((edge["from"], edge["output"]))
                fields = schemas[edge["output"]].get("fields", [])
                free_text = len(fields) == 1 and fields[0]["type"] == "string"
                cases.append((spec_path.stem.replace("-", "_"), edge["from"], edge["output"],
                              fields[0]["name"] if free_text else None))
    return cases


SCHEMA_OUTPUT_CASES = _schema_output_cases()
_PROSE = "I think the answer is 42, because six times seven is forty-two."


def test_schema_output_cases_cover_both_kinds():
    assert len(SCHEMA_OUTPUT_CASES) > 90, len(SCHEMA_OUTPUT_CASES)
    assert any(c[3] for c in SCHEMA_OUTPUT_CASES) and any(c[3] is None for c in SCHEMA_OUTPUT_CASES)
    assert ("kb_react", "think_or_act", "ReActStep", None) in SCHEMA_OUTPUT_CASES


def _call_step_with(m, backend, process, schema, target_reply):
    schema_names = set(m.SCHEMAS)

    def reply_for(schema_name):
        if schema_name == schema:
            return target_reply
        return json.dumps({f["name"]: _good_value(f["type"], schema_names) for f in m.SCHEMAS[schema_name]["fields"]})

    for name in [n for n in dir(m) if n.startswith("invoke_")]:
        setattr(m, name, lambda msg, output_schema=None: reply_for(output_schema or schema))
    seed = {}
    for s in m.SCHEMAS.values():
        for f in s["fields"]:
            seed.setdefault(f["name"], _good_value(f["type"], schema_names))
    seed.update(copy.deepcopy(_SEED))
    if backend == "custom":
        state = m.AgentState()
        state.data.update(seed)
        getattr(m, f"process_{process}")(state)
        return state.data
    return {**seed, **getattr(m, f"node_{process}")(dict(seed))}


@pytest.mark.parametrize("backend", ["custom", "langgraph"])
@pytest.mark.parametrize("spec_name,process,schema,free_text_field", SCHEMA_OUTPUT_CASES,
                         ids=[f"{c[0]}.{c[1]}:{c[2]}" for c in SCHEMA_OUTPUT_CASES])
def test_non_json_reply_is_loud_unless_schema_is_free_text(spec_name, process, schema, free_text_field,
                                                           backend, isolated, monkeypatch):
    folder, suffix = ("agents", "_agent") if backend == "custom" else ("agents_lg", "_agent_lg")
    m = _load(folder, f"{spec_name}{suffix}", monkeypatch)

    if free_text_field:
        after = _with_deadline(30, lambda: _call_step_with(m, backend, process, schema, _PROSE))
        assert after.get(free_text_field) == _PROSE, (free_text_field, after.get(free_text_field))
        with pytest.raises(m.SchemaViolationError):
            _with_deadline(30, lambda: _call_step_with(m, backend, process, schema, "   "))
        return

    with pytest.raises(m.SchemaViolationError) as err:
        _with_deadline(30, lambda: _call_step_with(m, backend, process, schema, _PROSE))
    assert schema in str(err.value) and _PROSE[:30] in str(err.value), str(err.value)


def test_answer_extraction_never_scores_the_question():
    """extract_answer's last-resort fallback concatenated every top-level string
    in state -- including the question the benchmark passed in. An agent that
    produced no answer at all then scored 1.0 on any ARC item whose key is A,
    because the question's "A) ..." was read as the chosen letter."""
    sys.path.insert(0, str(REPO_ROOT / "agent_ontology"))
    from benchmarks.scoring import extract_answer, score_arc

    question = "Which is a mammal? A) whale B) shark C) trout D) eel"
    state = {"question": question, "answer": "", "thought": ""}
    predicted = extract_answer(state, inputs={"question": question})
    assert question not in predicted, predicted
    assert score_arc(predicted, "A")["em"] == 0.0
    # The fallback still finds an answer the agent put under an unlisted key.
    assert extract_answer({"question": question, "verdict": "A"}, inputs={"question": question}) == "A"


# ── 9. A JSON object with none of the schema's fields is not a reply ─────
#
# After section 8, {"unrelated": ...} was still accepted: every declared field
# was "omitted", reset to its default, and the step carried on as though the
# model had answered. The rule: when the schema has any required
# (non-optional) field and the reply carries none of the declared fields, it
# raises SchemaViolationError. A reply carrying only an optional declared field
# (BabyAGI's tool_calls-only execution reply) is still a reply.


def _required_field_cases():
    import yaml

    out = []
    for spec_name, process, schema, _ in SCHEMA_OUTPUT_CASES:
        path = REPO_ROOT / "agent_ontology" / "specs" / f"{spec_name}.yaml"
        if not path.exists():
            path = REPO_ROOT / "agent_ontology" / "specs" / f"{spec_name.replace('_', '-')}.yaml"
        spec = yaml.safe_load(path.read_text())
        fields = next(s for s in spec["schemas"] if s["name"] == schema).get("fields", [])
        if any(not f.get("optional") for f in fields):
            out.append((spec_name, process, schema))
    return out


REQUIRED_FIELD_CASES = _required_field_cases()


@pytest.mark.parametrize("backend", ["custom", "langgraph"])
@pytest.mark.parametrize("spec_name,process,schema", REQUIRED_FIELD_CASES,
                         ids=[f"{c[0]}.{c[1]}:{c[2]}" for c in REQUIRED_FIELD_CASES])
def test_object_with_no_declared_field_is_loud(spec_name, process, schema, backend, isolated, monkeypatch):
    folder, suffix = ("agents", "_agent") if backend == "custom" else ("agents_lg", "_agent_lg")
    m = _load(folder, f"{spec_name}{suffix}", monkeypatch)
    reply = json.dumps({"_unrelated_note": "I could not decide", "_unrelated_score": 0.2})
    with pytest.raises(m.SchemaViolationError) as err:
        _with_deadline(30, lambda: _call_step_with(m, backend, process, schema, reply))
    assert schema in str(err.value) and "I could not decide" in str(err.value), str(err.value)


def test_optional_only_reply_is_still_a_reply(isolated, monkeypatch):
    """BabyAGI's execution agent may answer with only its optional tool_calls."""
    m = _load("agents", "babyagi_agent", monkeypatch)
    fields = {f["name"]: f for f in m.SCHEMAS["ExecutionOutput"]["fields"]}
    assert fields["tool_calls"].get("optional") and not fields["result"].get("optional")
    assert m.require_valid_output({"tool_calls": []}, "ExecutionOutput", "Execution Agent") == {"tool_calls": []}
    with pytest.raises(m.SchemaViolationError):
        m.require_valid_output({"note": "x"}, "ExecutionOutput", "Execution Agent")


