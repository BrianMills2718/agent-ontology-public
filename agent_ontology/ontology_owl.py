#!/usr/bin/env python3
"""
Agent Ontology Formal Ontology — OWL/DL Prototype

Translates the Agent Ontology into OWL using owlready2,
loads YAML specs as OWL instances, and uses a DL reasoner to
automatically classify agent architectures by structural patterns.

This is Phase A of the formal ontology roadmap:
- Define core types as OWL classes
- Define architectural patterns as DL concepts (necessary & sufficient conditions)
- Load specs as instances
- Run reasoner to auto-classify

Usage:
  python3 ontology_owl.py                    # Classify all specs
  python3 ontology_owl.py specs/react.yaml   # Classify one spec
  python3 ontology_owl.py --dump-ontology    # Print ontology structure
"""

import sys
import os
import yaml
from owlready2 import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
# 1. BUILD THE OWL ONTOLOGY
# ═══════════════════════════════════════════════════════════════

def build_ontology():
    """Create the Agent Ontology in OWL."""
    onto = get_ontology("http://agent-ontology.org/ontology#")

    with onto:
        # ── Top-level classes ──────────────────────────────────
        class Spec(Thing):
            """An agent architecture specification."""
            pass

        class Node(Thing):
            """Any node in the architecture graph."""
            pass

        class Entity(Node):
            """A persistent thing (agent, store, tool, etc.)."""
            pass

        class Process(Node):
            """A reified relationship (step, gate, checkpoint, etc.)."""
            pass

        class Edge(Thing):
            """A relationship between nodes."""
            pass

        # ── Entity subclasses ──────────────────────────────────
        class Agent(Entity):
            """An LLM-based reasoning unit."""
            pass

        class Store(Entity):
            """A persistence layer."""
            pass

        class Tool(Entity):
            """An external capability."""
            pass

        class Human(Entity):
            """A human participant."""
            pass

        class Channel(Entity):
            """A pub/sub communication channel."""
            pass

        class Team(Entity):
            """A group of agents with execution strategy."""
            pass

        class Conversation(Entity):
            """A multi-turn dialogue."""
            pass

        class Config(Entity):
            """A configuration object."""
            pass

        # ── Process subclasses ─────────────────────────────────
        class Step(Process):
            """A computation/orchestration point."""
            pass

        class Gate(Process):
            """A decision/branch point."""
            pass

        class Checkpoint(Process):
            """A human-in-the-loop pause."""
            pass

        class Spawn(Process):
            """Dynamic agent instantiation."""
            pass

        class Protocol(Process):
            """A multi-party interaction pattern."""
            pass

        class Policy(Process):
            """A cross-cutting behavioral rule."""
            pass

        class ErrorHandler(Process):
            """Structured error handling."""
            pass

        # ── Object Properties (relationships) ──────────────────
        # Spec -> nodes
        class has_entity(Spec >> Entity):
            pass

        class has_process(Spec >> Process):
            pass

        # Process -> Process flow
        class flows_to(Process >> Process):
            """Control flow from one process to another."""
            pass

        class loops_to(Process >> Process):
            """Loop back-edge."""
            pass

        class branches_to(Gate >> Process):
            """Conditional branch from gate."""
            pass

        # Process -> Entity invocations
        class invokes(Process >> Entity):
            """A process calls an entity."""
            pass

        # Store access
        class reads_from(Process >> Store):
            """A process reads from a store."""
            pass

        class writes_to(Process >> Store):
            """A process writes to a store."""
            pass

        # Channel pub/sub
        class publishes_to(Process >> Channel):
            """A process publishes to a channel."""
            pass

        class subscribes_from(Process >> Channel):  # note: reversed direction
            """A process subscribes from a channel."""
            pass

        # Agent handoff
        class hands_off_to(Agent >> Agent):
            """Agent-to-agent control transfer."""
            pass

        # Data properties
        class has_label(Node >> str):
            pass

        class has_model(Agent >> str):
            pass

        class has_condition(Gate >> str):
            pass

        class has_logic(Step >> str):
            pass

        class has_store_type(Store >> str):
            pass

        class has_tool_type(Tool >> str):
            pass

        class has_strategy(Team >> str):
            pass

        # ── PATTERN DEFINITIONS (DL concepts) ──────────────────
        #
        # This is where the magic happens. We define architectural
        # patterns as Description Logic concepts using necessary
        # and sufficient conditions. The reasoner will automatically
        # classify any spec that matches the structural definition.

        # Pattern: Reasoning Loop (ReAct-like)
        # Definition: A spec that has a step which invokes an agent,
        # flows to a gate, and the gate loops back to the step.
        class ReasoningLoopSpec(Spec):
            """A spec with a reason-act-observe loop pattern."""
            equivalent_to = [
                Spec
                & has_process.some(
                    Step
                    & invokes.some(Agent)
                    & flows_to.some(Gate & loops_to.some(Step))
                )
            ]

        # Pattern: Critique Cycle (Self-Refine-like)
        # Definition: A spec with TWO steps where one invokes a generator
        # and another invokes a critic, connected by a gate that can loop.
        class CritiqueCycleSpec(Spec):
            """A spec with a generate-critique-refine loop pattern."""
            equivalent_to = [
                Spec
                & has_process.some(Step & invokes.some(Agent) & flows_to.some(Step & invokes.some(Agent)))
                & has_process.some(Gate & loops_to.some(Step))
            ]

        # Pattern: Retrieval-Augmented (RAG-like)
        # Definition: A spec with a step that reads from a store,
        # and another step that invokes a generator agent.
        class RetrievalAugmentedSpec(Spec):
            """A spec with a retrieve-then-generate pattern."""
            equivalent_to = [
                Spec
                & has_process.some(Step & reads_from.some(Store))
                & has_process.some(Step & invokes.some(Agent))
            ]

        # Pattern: Fan-Out (parallel invocation)
        # Definition: A spec with a single step that invokes
        # multiple different agents.
        class FanOutSpec(Spec):
            """A spec with parallel agent invocations from one step."""
            equivalent_to = [
                Spec
                & has_process.some(
                    Step
                    & invokes.min(2, Agent)
                )
            ]

        # Pattern: Human-in-the-Loop
        # Definition: A spec with a checkpoint process.
        class HumanInLoopSpec(Spec):
            """A spec with human oversight checkpoints."""
            equivalent_to = [
                Spec
                & has_process.some(Checkpoint)
            ]

        # Pattern: Multi-Agent Debate
        # Definition: A spec with 3+ agents and a gate that loops.
        class MultiAgentDebateSpec(Spec):
            """A spec with multiple agents in a debate/discussion pattern."""
            equivalent_to = [
                Spec
                & has_entity.min(3, Agent)
                & has_process.some(Gate & loops_to.some(Step))
            ]

        # Pattern: Pub/Sub Communication
        # Definition: A spec with channels and publish/subscribe edges.
        class PubSubSpec(Spec):
            """A spec using channel-based pub/sub communication."""
            equivalent_to = [
                Spec
                & has_entity.some(Channel)
                & has_process.some(Step & publishes_to.some(Channel))
            ]

        # Pattern: Handoff-Based Routing
        # Definition: A spec with handoff edges between agents.
        class HandoffSpec(Spec):
            """A spec using agent-to-agent handoffs."""
            equivalent_to = [
                Spec
                & has_entity.some(Agent & hands_off_to.some(Agent))
            ]

        # Pattern: Store-Backed Memory
        # Definition: A spec with store reads AND writes (not just one-way).
        class MemoryBackedSpec(Spec):
            """A spec that reads from and writes to persistent stores."""
            equivalent_to = [
                Spec
                & has_process.some(Step & reads_from.some(Store))
                & has_process.some(Step & writes_to.some(Store))
            ]

    return onto


# ═══════════════════════════════════════════════════════════════
# 2. LOAD YAML SPECS AS OWL INSTANCES
# ═══════════════════════════════════════════════════════════════

def load_spec_as_instances(onto, spec_path):
    """Load a YAML spec and create OWL instances."""
    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    if not spec:
        return None

    # Use filename as unique ID (spec names can collide, e.g. two "BabyAGI" specs)
    file_id = os.path.basename(spec_path).replace(".yaml", "")
    spec_name = spec.get("name", file_id)
    safe_name = file_id.replace(" ", "_").replace("-", "_").replace(".", "_")

    with onto:
        # Create spec instance
        spec_inst = onto.Spec(f"spec_{safe_name}")

        # Entity map for edge resolution
        entity_instances = {}

        # Create entity instances
        for e in spec.get("entities", []):
            eid = e["id"]
            etype = e["type"]
            safe_eid = f"{safe_name}_{eid}"

            # Map type to OWL class
            cls_map = {
                "agent": onto.Agent,
                "store": onto.Store,
                "tool": onto.Tool,
                "human": onto.Human,
                "channel": onto.Channel,
                "team": onto.Team,
                "conversation": onto.Conversation,
                "config": onto.Config,
            }
            cls = cls_map.get(etype, onto.Entity)
            inst = cls(safe_eid)
            inst.has_label = [e.get("label", eid)]

            if etype == "agent" and "model" in e:
                inst.has_model = [e["model"]]
            elif etype == "store" and "store_type" in e:
                inst.has_store_type = [e["store_type"]]
            elif etype == "tool" and "tool_type" in e:
                inst.has_tool_type = [e["tool_type"]]
            elif etype == "team" and "strategy" in e:
                inst.has_strategy = [e["strategy"]]

            spec_inst.has_entity.append(inst)
            entity_instances[eid] = inst

        # Create process instances
        process_instances = {}
        for p in spec.get("processes", []):
            pid = p["id"]
            ptype = p["type"]
            safe_pid = f"{safe_name}_{pid}"

            cls_map = {
                "step": onto.Step,
                "gate": onto.Gate,
                "checkpoint": onto.Checkpoint,
                "spawn": onto.Spawn,
                "protocol": onto.Protocol,
                "policy": onto.Policy,
                "error_handler": onto.ErrorHandler,
            }
            cls = cls_map.get(ptype, onto.Process)
            inst = cls(safe_pid)
            inst.has_label = [p.get("label", pid)]

            if ptype == "gate" and "condition" in p:
                inst.has_condition = [p["condition"]]
            if ptype == "step" and "logic" in p:
                inst.has_logic = [p["logic"][:200]]  # truncate

            spec_inst.has_process.append(inst)
            process_instances[pid] = inst

        # Create edge relationships
        for e in spec.get("edges", []):
            etype = e["type"]
            src = e.get("from", "")
            tgt = e.get("to", "")
            src_inst = process_instances.get(src) or entity_instances.get(src)
            tgt_inst = process_instances.get(tgt) or entity_instances.get(tgt)

            if not src_inst or not tgt_inst:
                continue

            if etype == "flow" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Process):
                src_inst.flows_to.append(tgt_inst)
            elif etype == "invoke" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Entity):
                src_inst.invokes.append(tgt_inst)
            elif etype == "loop" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Process):
                src_inst.loops_to.append(tgt_inst)
            elif etype == "branch" and isinstance(src_inst, onto.Gate) and isinstance(tgt_inst, onto.Process):
                src_inst.branches_to.append(tgt_inst)
            elif etype == "read" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Store):
                src_inst.reads_from.append(tgt_inst)
            elif etype == "write" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Store):
                src_inst.writes_to.append(tgt_inst)
            elif etype == "publish" and isinstance(src_inst, onto.Process) and isinstance(tgt_inst, onto.Channel):
                src_inst.publishes_to.append(tgt_inst)
            elif etype == "subscribe" and isinstance(tgt_inst, onto.Process) and isinstance(src_inst, onto.Channel):
                tgt_inst.subscribes_from.append(src_inst)
            elif etype == "handoff" and isinstance(src_inst, onto.Agent) and isinstance(tgt_inst, onto.Agent):
                src_inst.hands_off_to.append(tgt_inst)

        # Compute process ordering for backward-edge detection.
        # Use the YAML process definition order as the canonical ordering.
        # Specs are written with processes in topological order, and this
        # avoids issues with BFS stalling at entity invocations.
        proc_order = {}
        for idx, p in enumerate(spec.get("processes", [])):
            proc_order[p["id"]] = idx

        # Handle gate branches from process definitions
        for p in spec.get("processes", []):
            if p["type"] == "gate":
                pid = p["id"]
                gate_inst = process_instances.get(pid)
                if gate_inst:
                    for branch in p.get("branches", []):
                        target_id = branch.get("target", "")
                        target_inst = process_instances.get(target_id)
                        if target_inst and target_inst not in gate_inst.branches_to:
                            gate_inst.branches_to.append(target_inst)
                    # Only mark as loops_to if target appears BEFORE gate in flow order
                    # (i.e., it's a backward edge creating a cycle)
                    gate_order = proc_order.get(pid, 999)
                    for branch in p.get("branches", []):
                        target_id = branch.get("target", "")
                        target_inst = process_instances.get(target_id)
                        target_order = proc_order.get(target_id, 999)
                        if target_inst and isinstance(target_inst, onto.Step) and target_order < gate_order:
                            if target_inst not in gate_inst.loops_to:
                                gate_inst.loops_to.append(target_inst)

        # Also detect backward FLOW edges (not just gate branches).
        # e.g., code_reviewer: wait_for_new_commits → intake_pr is a flow edge
        # that creates a cycle.
        for edge in spec.get("edges", []):
            if edge["type"] == "flow":
                src_id = edge.get("from", "")
                tgt_id = edge.get("to", "")
                src_order = proc_order.get(src_id, 999)
                tgt_order = proc_order.get(tgt_id, 999)
                if tgt_order < src_order:  # backward edge
                    src_inst = process_instances.get(src_id)
                    tgt_inst = process_instances.get(tgt_id)
                    if src_inst and tgt_inst and isinstance(tgt_inst, onto.Step):
                        if tgt_inst not in src_inst.loops_to:
                            src_inst.loops_to.append(tgt_inst)

    return spec_inst


# ═══════════════════════════════════════════════════════════════
# 3. RUN REASONER AND REPORT
# ═══════════════════════════════════════════════════════════════

def classify_specs(onto, spec_instances):
    """Run the reasoner and report pattern classifications."""

    import shutil
    java_available = shutil.which("java") is not None

    if java_available:
        print("Running DL reasoner...")
        print()
        try:
            with onto:
                sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
        except Exception:
            try:
                with onto:
                    sync_reasoner_hermit(infer_property_values=True)
            except Exception as ex:
                print(f"  Warning: Reasoner failed ({ex}), using structural matching fallback")
                return classify_structural(onto, spec_instances)
    else:
        print("Java not available — using structural pattern matching.")
        print("(Install JRE for Pellet/HermiT DL reasoning.)")
        print()
        return classify_structural(onto, spec_instances)

    # Pattern classes to check
    pattern_classes = [
        ("ReasoningLoop", onto.ReasoningLoopSpec),
        ("CritiqueCycle", onto.CritiqueCycleSpec),
        ("RetrievalAugmented", onto.RetrievalAugmentedSpec),
        ("FanOut", onto.FanOutSpec),
        ("HumanInLoop", onto.HumanInLoopSpec),
        ("MultiAgentDebate", onto.MultiAgentDebateSpec),
        ("PubSub", onto.PubSubSpec),
        ("Handoff", onto.HandoffSpec),
        ("MemoryBacked", onto.MemoryBackedSpec),
    ]

    results = {}
    for spec_inst in spec_instances:
        name = spec_inst.name.replace("spec_", "")
        patterns = []
        for pname, pcls in pattern_classes:
            if isinstance(spec_inst, pcls) or pcls in spec_inst.is_a:
                patterns.append(pname)
        results[name] = patterns

    return results


def classify_structural(onto, spec_instances):
    """Fallback: structural pattern matching without reasoner.

    Pattern definitions are intentionally tight to avoid over-classification.
    Each pattern checks for the distinguishing structural signature, not just
    necessary conditions.
    """
    results = {}

    for spec_inst in spec_instances:
        name = spec_inst.name.replace("spec_", "")
        patterns = []

        processes = list(spec_inst.has_process)
        entities = list(spec_inst.has_entity)

        steps = [p for p in processes if isinstance(p, onto.Step)]
        gates = [p for p in processes if isinstance(p, onto.Gate)]
        checkpoints = [p for p in processes if isinstance(p, onto.Checkpoint)]

        agents = [e for e in entities if isinstance(e, onto.Agent)]
        stores = [e for e in entities if isinstance(e, onto.Store)]
        channels = [e for e in entities if isinstance(e, onto.Channel)]

        # ── Helpers ─────────────────────────────────────────────────

        # Build forward-flow adjacency for BFS
        flow_adj = {}  # node -> list of flow successors
        for p in processes:
            flow_adj[p] = []
            for tgt in p.flows_to:
                flow_adj[p].append(tgt)
            if isinstance(p, onto.Gate):
                for tgt in p.branches_to:
                    if tgt not in flow_adj[p]:
                        flow_adj[p].append(tgt)
                for tgt in p.loops_to:
                    if tgt not in flow_adj[p]:
                        flow_adj[p].append(tgt)

        def bfs_steps_in_loop(entry_step):
            """BFS forward from entry, collect all steps until we loop back."""
            visited = set()
            queue = [entry_step]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                for succ in flow_adj.get(node, []):
                    if succ not in visited:
                        queue.append(succ)
            return [n for n in visited if isinstance(n, onto.Step)]

        # Collect ALL loop targets from any process (not just gates — loop edges
        # can originate from steps too, e.g. self_refine: refine --loop--> generate)
        all_loop_targets = set()  # direct targets of backward edges
        for p in processes:
            for tgt in p.loops_to:
                all_loop_targets.add(tgt)
        for g in gates:
            for tgt in g.loops_to:
                all_loop_targets.add(tgt)

        # Compute full loop body: all steps reachable from any loop target via BFS.
        # This captures steps INSIDE a loop, not just the loop entry point.
        # e.g., in LATS: select_node is the target, but expand_node and
        # evaluate_node (which invoke agents) are in the loop body too.
        loop_body = set()
        for target in all_loop_targets:
            for node in bfs_steps_in_loop(target):
                loop_body.add(node)

        # ── ReasoningLoop ──────────────────────────────────────────
        # A step S invokes an agent and is within a loop body.
        for s in steps:
            if not any(isinstance(i, onto.Agent) for i in s.invokes):
                continue
            if s in loop_body:
                patterns.append("ReasoningLoop")
                break

        # ── CritiqueCycle ──────────────────────────────────────────
        # Sequential generate→critique chain: step A flows_to step B,
        # BOTH invoke agents (different roles), BOTH are in a loop body,
        # and there's a gate downstream of B that loops back toward A.
        # The A→B direct flow distinguishes this from debate.
        for sa in steps:
            if sa not in loop_body:
                continue
            if not any(isinstance(i, onto.Agent) for i in sa.invokes):
                continue
            for sb_or_g in sa.flows_to:
                if not isinstance(sb_or_g, onto.Step):
                    continue
                if sb_or_g not in loop_body:
                    continue
                if not any(isinstance(i, onto.Agent) for i in sb_or_g.invokes):
                    continue
                # Both A and B invoke agents and are in a loop body.
                # Additionally require that A is reachable via a backward edge
                # (i.e., A is an actual loop target, meaning the loop restarts at A).
                if sa in all_loop_targets:
                    if "CritiqueCycle" not in patterns:
                        patterns.append("CritiqueCycle")

        # ── RetrievalAugmented ─────────────────────────────────────
        # A step reads from a store AND a (possibly different) step invokes
        # an agent. The read must feed into generation.
        reading_steps = [s for s in steps if s.reads_from]
        invoking_steps = [s for s in steps if any(isinstance(i, onto.Agent) for i in s.invokes)]
        if reading_steps and invoking_steps:
            patterns.append("RetrievalAugmented")

        # ── FanOut ─────────────────────────────────────────────────
        # A single step invokes 2+ agents (parallel dispatch).
        if any(len([i for i in s.invokes if isinstance(i, onto.Agent)]) >= 2 for s in steps):
            patterns.append("FanOut")

        # ── HumanInLoop ────────────────────────────────────────────
        # Has checkpoint process.
        if checkpoints:
            patterns.append("HumanInLoop")

        # ── MultiAgentDebate ───────────────────────────────────────
        # 3+ agents that are each invoked by DIFFERENT steps within a loop.
        # The key: multiple distinct agents participate in the looping subgraph,
        # not just 3 agents existing in the spec. Uses BFS to find full loop body.
        if len(agents) >= 3:
            loop_agents = set()
            for target in all_loop_targets:
                if isinstance(target, onto.Step):
                    body_steps = bfs_steps_in_loop(target)
                    for s in body_steps:
                        for inv in s.invokes:
                            if isinstance(inv, onto.Agent):
                                loop_agents.add(inv)
            if len(loop_agents) >= 3:
                patterns.append("MultiAgentDebate")

        # ── PubSub ─────────────────────────────────────────────────
        # Has channels AND processes that publish to them.
        if channels and any(s.publishes_to for s in steps):
            patterns.append("PubSub")

        # ── Handoff ────────────────────────────────────────────────
        # Agent-to-agent handoff edges.
        if any(a.hands_off_to for a in agents):
            patterns.append("Handoff")

        # ── MemoryBacked ───────────────────────────────────────────
        # Store reads AND writes (bidirectional persistence).
        if any(s.reads_from for s in steps) and any(s.writes_to for s in steps):
            patterns.append("MemoryBacked")

        # Deduplicate
        results[name] = list(dict.fromkeys(patterns))

    return results


def print_results(results):
    """Pretty-print classification results."""
    print("=" * 70)
    print("  OWL/DL Pattern Classification Results")
    print("=" * 70)
    print()

    # Summary table
    all_patterns = set()
    for patterns in results.values():
        all_patterns.update(patterns)
    all_patterns = sorted(all_patterns)

    # Header
    name_width = max(len(n) for n in results) + 2
    print(f"  {'Spec':<{name_width}} Patterns")
    print(f"  {'─' * name_width} {'─' * 50}")

    for name, patterns in sorted(results.items()):
        pstr = ", ".join(patterns) if patterns else "(no patterns detected)"
        print(f"  {name:<{name_width}} {pstr}")

    print()

    # Pattern coverage matrix
    print("Pattern Coverage Matrix:")
    print()
    for pattern in all_patterns:
        specs_with = [n for n, ps in results.items() if pattern in ps]
        print(f"  {pattern:<25} ({len(specs_with)} specs): {', '.join(sorted(specs_with))}")

    print()
    print(f"  Total: {len(results)} specs, {len(all_patterns)} distinct patterns")


def dump_ontology(onto):
    """Print ontology structure."""
    print("=" * 70)
    print("  Agent Ontology OWL Structure")
    print("=" * 70)
    print()

    print("Classes:")
    for cls in onto.classes():
        parents = [p.name for p in cls.is_a if hasattr(p, 'name')]
        equiv = cls.equivalent_to
        print(f"  {cls.name}")
        if parents:
            print(f"    parents: {', '.join(parents)}")
        if equiv:
            print(f"    equivalent_to: {equiv}")
    print()

    print("Object Properties:")
    for prop in onto.object_properties():
        domain = [d.name for d in prop.domain] if prop.domain else ["?"]
        range_ = [r.name for r in prop.range] if prop.range else ["?"]
        print(f"  {prop.name}: {', '.join(domain)} -> {', '.join(range_)}")
    print()

    print("Data Properties:")
    for prop in onto.data_properties():
        print(f"  {prop.name}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import glob

    if "--dump-ontology" in sys.argv:
        onto = build_ontology()
        dump_ontology(onto)
        return

    # Build ontology
    onto = build_ontology()

    # Load specs
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        spec_files = sys.argv[1:]
    else:
        spec_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "specs", "*.yaml")))

    print(f"Loading {len(spec_files)} specs into OWL ontology...")
    print()

    spec_instances = []
    for sf in spec_files:
        name = os.path.basename(sf)
        try:
            inst = load_spec_as_instances(onto, sf)
            if inst:
                spec_instances.append(inst)
                entities = list(inst.has_entity)
                processes = list(inst.has_process)
                print(f"  {name:<40} {len(entities)} entities, {len(processes)} processes")
        except Exception as ex:
            print(f"  {name:<40} FAILED: {ex}")

    print()

    # Classify
    results = classify_specs(onto, spec_instances)

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
