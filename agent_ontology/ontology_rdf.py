#!/usr/bin/env python3
"""
RDF/SPARQL Experiment — Phase A.5

Question: Does modeling specs as RDF with reified edges and querying with
schema semantics solve problems that structural matching can't?

Answer: YES. The CritiqueCycle false positive (BabyAGI) is correctly excluded
when we check output schema field names for evaluation-related keywords.
Self-Refine's CriticOutput has "quality_score", "weaknesses", "specific_feedback".
BabyAGI's EnrichedResult has "result", "context" — no evaluation fields.

The signal was already in the YAML schemas. Reified edges give us a data model
where this information is queryable. No manual role annotation needed.

Note: rdflib's SPARQL engine is too slow for multi-join pattern queries
(>5min on 695 triples). We use Python graph traversal over the RDF data model
instead. A real triplestore (Jena Fuseki, Blazegraph) would handle SPARQL fine.

Usage:
  python3 ontology_rdf.py                    # All specs
  python3 ontology_rdf.py specs/react.yaml   # Specific specs
  python3 ontology_rdf.py --dump             # Print Turtle RDF
"""

import os
import sys
import yaml
import glob
import time
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OC = Namespace("http://agent-ontology.org/ontology#")

# Evaluation-related field name keywords
EVAL_KEYWORDS = {"score", "quality", "feedback", "weakness", "error",
                 "rating", "evaluation", "critique", "improvement"}


# ═══════════════════════════════════════════════════════════════
# 1. EXPORT YAML SPEC TO RDF WITH REIFIED EDGES
# ═══════════════════════════════════════════════════════════════

def spec_to_rdf(g, spec_path):
    """Export a YAML spec to RDF triples with reified edges."""
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    if not spec:
        return None

    file_id = os.path.basename(spec_path).replace(".yaml", "")
    spec_uri = OC[f"spec_{file_id}"]
    g.add((spec_uri, RDF.type, OC.Spec))
    g.add((spec_uri, RDFS.label, Literal(spec.get("name", file_id))))

    # ── Schemas with fields ─────────────────────────────────
    schema_map = {}
    for s in spec.get("schemas", []):
        sname = s["name"]
        schema_uri = OC[f"{file_id}_schema_{sname}"]
        g.add((schema_uri, RDF.type, OC.Schema))
        g.add((schema_uri, RDFS.label, Literal(sname)))
        g.add((spec_uri, OC.hasSchema, schema_uri))
        schema_map[sname] = schema_uri

        for field in s.get("fields", []):
            if isinstance(field, dict):
                fname = field.get("name", "")
                ftype = field.get("type", "")
            else:
                fname = str(field)
                ftype = ""
            if fname:
                field_uri = OC[f"{file_id}_field_{sname}_{fname}"]
                g.add((field_uri, RDF.type, OC.Field))
                g.add((field_uri, RDFS.label, Literal(fname)))
                g.add((field_uri, OC.fieldType, Literal(ftype)))
                g.add((schema_uri, OC.hasField, field_uri))

    # ── Entities ────────────────────────────────────────────
    entity_uris = {}
    type_map = {
        "agent": OC.Agent, "store": OC.Store, "tool": OC.Tool,
        "human": OC.Human, "channel": OC.Channel, "team": OC.Team,
        "conversation": OC.Conversation, "config": OC.Config,
    }
    for e in spec.get("entities", []):
        eid = e["id"]
        uri = OC[f"{file_id}_{eid}"]
        g.add((uri, RDF.type, type_map.get(e["type"], OC.Entity)))
        g.add((uri, RDFS.label, Literal(e.get("label", eid))))
        g.add((spec_uri, OC.hasEntity, uri))
        entity_uris[eid] = uri

    # ── Processes with data_in/data_out ─────────────────────
    proc_uris = {}
    proc_type_map = {
        "step": OC.Step, "gate": OC.Gate, "checkpoint": OC.Checkpoint,
        "spawn": OC.Spawn, "protocol": OC.Protocol, "policy": OC.Policy,
        "error_handler": OC.ErrorHandler,
    }
    proc_order = {}
    for idx, p in enumerate(spec.get("processes", [])):
        pid = p["id"]
        uri = OC[f"{file_id}_{pid}"]
        g.add((uri, RDF.type, proc_type_map.get(p["type"], OC.Process)))
        g.add((uri, RDFS.label, Literal(p.get("label", pid))))
        g.add((spec_uri, OC.hasProcess, uri))
        proc_uris[pid] = uri
        proc_order[pid] = idx

        if p.get("data_in") and p["data_in"] in schema_map:
            g.add((uri, OC.inputSchema, schema_map[p["data_in"]]))
        if p.get("data_out") and p["data_out"] in schema_map:
            g.add((uri, OC.outputSchema, schema_map[p["data_out"]]))

    # ── Reified edges ───────────────────────────────────────
    edge_cls_map = {
        "flow": OC.FlowEdge, "invoke": OC.InvokeEdge,
        "loop": OC.LoopEdge, "branch": OC.BranchEdge,
        "read": OC.ReadEdge, "write": OC.WriteEdge,
        "publish": OC.PublishEdge, "subscribe": OC.SubscribeEdge,
        "handoff": OC.HandoffEdge, "error": OC.ErrorEdge,
    }
    for i, e in enumerate(spec.get("edges", [])):
        edge_uri = OC[f"{file_id}_edge_{i}"]
        src_uri = proc_uris.get(e.get("from")) or entity_uris.get(e.get("from"))
        tgt_uri = proc_uris.get(e.get("to")) or entity_uris.get(e.get("to"))
        if not src_uri or not tgt_uri:
            continue

        g.add((edge_uri, RDF.type, edge_cls_map.get(e["type"], OC.Edge)))
        g.add((edge_uri, OC.fromNode, src_uri))
        g.add((edge_uri, OC.toNode, tgt_uri))
        g.add((spec_uri, OC.hasEdge, edge_uri))
        if e.get("label"):
            g.add((edge_uri, RDFS.label, Literal(e["label"])))
        if e.get("data") and e["data"] in schema_map:
            g.add((edge_uri, OC.dataSchema, schema_map[e["data"]]))

        # Mark backward flow edges
        if e["type"] == "flow":
            src_order = proc_order.get(e.get("from"), 999)
            tgt_order = proc_order.get(e.get("to"), 999)
            if tgt_order < src_order:
                g.add((edge_uri, RDF.type, OC.BackwardFlowEdge))

    # ── Gate branches as reified edges ──────────────────────
    for p in spec.get("processes", []):
        if p["type"] == "gate":
            gate_uri = proc_uris.get(p["id"])
            gate_order = proc_order.get(p["id"], 999)
            for j, branch in enumerate(p.get("branches", [])):
                target_uri = proc_uris.get(branch.get("target"))
                if gate_uri and target_uri:
                    edge_uri = OC[f"{file_id}_branch_{p['id']}_{j}"]
                    g.add((edge_uri, RDF.type, OC.BranchEdge))
                    g.add((edge_uri, OC.fromNode, gate_uri))
                    g.add((edge_uri, OC.toNode, target_uri))
                    g.add((spec_uri, OC.hasEdge, edge_uri))
                    target_order = proc_order.get(branch.get("target"), 999)
                    if target_order < gate_order:
                        g.add((edge_uri, RDF.type, OC.BackwardBranchEdge))

    return spec_uri


# ═══════════════════════════════════════════════════════════════
# 2. PYTHON GRAPH QUERIES OVER RDF DATA MODEL
# ═══════════════════════════════════════════════════════════════

def query_critique_cycle(g):
    """Find CritiqueCycle pattern at three levels: structural, semantic, data-flow.

    Returns dict: spec_name -> {structural, semantic, dataflow, details}
    """
    results = {}

    for spec in g.subjects(RDF.type, OC.Spec):
        spec_name = str(spec).split("spec_")[-1]

        # Check for any backward edge (loop exists)
        has_loop = False
        for edge in g.objects(spec, OC.hasEdge):
            if ((edge, RDF.type, OC.LoopEdge) in g or
                (edge, RDF.type, OC.BackwardFlowEdge) in g or
                (edge, RDF.type, OC.BackwardBranchEdge) in g):
                has_loop = True
                break
        if not has_loop:
            continue

        # Find invoke edges: step -> agent
        invoke_steps = {}
        for edge in g.objects(spec, OC.hasEdge):
            if (edge, RDF.type, OC.InvokeEdge) not in g:
                continue
            from_node = next(g.objects(edge, OC.fromNode), None)
            to_node = next(g.objects(edge, OC.toNode), None)
            if from_node and to_node:
                if (from_node, RDF.type, OC.Step) in g and (to_node, RDF.type, OC.Agent) in g:
                    invoke_steps[from_node] = to_node

        # Find flow edges between two invoking steps
        for edge in g.objects(spec, OC.hasEdge):
            if (edge, RDF.type, OC.FlowEdge) not in g:
                continue
            step_a = next(g.objects(edge, OC.fromNode), None)
            step_b = next(g.objects(edge, OC.toNode), None)
            if step_a not in invoke_steps or step_b not in invoke_steps:
                continue
            if invoke_steps[step_a] == invoke_steps[step_b]:
                continue

            # STRUCTURAL MATCH: two different agent-invoking steps in sequence + loop
            a_label = str(next(g.objects(step_a, RDFS.label), "?"))
            b_label = str(next(g.objects(step_b, RDFS.label), "?"))

            # SEMANTIC CHECK: step B's output schema has evaluation fields
            b_out_fields = _get_schema_fields(g, step_b, OC.outputSchema)
            eval_fields = [f for f in b_out_fields
                           if any(kw in f.lower() for kw in EVAL_KEYWORDS)]

            # DATA-FLOW CHECK: step A's output fields overlap with step B's input fields
            a_out_fields = _get_schema_fields(g, step_a, OC.outputSchema)
            b_in_fields = _get_schema_fields(g, step_b, OC.inputSchema)
            shared_fields = set(a_out_fields) & set(b_in_fields)

            results[spec_name] = {
                "structural": True,
                "semantic": len(eval_fields) > 0,
                "dataflow": len(shared_fields) > 0 and len(eval_fields) > 0,
                "step_a": a_label,
                "step_b": b_label,
                "b_out_fields": b_out_fields,
                "eval_fields": eval_fields,
                "a_out_fields": a_out_fields,
                "b_in_fields": b_in_fields,
                "shared_fields": shared_fields,
            }
            break  # take first match per spec

    return results


def _get_schema_fields(g, step, schema_prop):
    """Get field names from a step's input or output schema."""
    schema = next(g.objects(step, schema_prop), None)
    if not schema:
        return []
    return [str(lbl) for field in g.objects(schema, OC.hasField)
            for lbl in g.objects(field, RDFS.label)]


# ═══════════════════════════════════════════════════════════════
# 3. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    g = Graph()
    g.bind("oc", OC)

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        spec_files = [a for a in sys.argv[1:] if not a.startswith("--")]
    else:
        spec_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, "specs", "*.yaml")))

    print(f"Exporting {len(spec_files)} specs to RDF...")
    for sf in spec_files:
        spec_to_rdf(g, sf)
    print(f"  {len(g)} triples")
    print()

    if "--dump" in sys.argv:
        print(g.serialize(format="turtle"))
        return

    # Run CritiqueCycle experiment
    t0 = time.time()
    results = query_critique_cycle(g)
    elapsed = time.time() - t0

    print("=" * 70)
    print("  CritiqueCycle Detection: Structural vs Semantic vs Data-flow")
    print("=" * 70)
    print()

    # Show detailed results for matches
    for name, r in sorted(results.items()):
        markers = []
        if r["structural"]:
            markers.append("STRUCTURAL")
        if r["semantic"]:
            markers.append("SEMANTIC")
        if r["dataflow"]:
            markers.append("DATAFLOW")

        print(f"  {name}")
        print(f"    {r['step_a']} -> {r['step_b']}")
        print(f"    B output fields: {r['b_out_fields']}")
        if r["eval_fields"]:
            print(f"    Eval fields:     {r['eval_fields']}")
        if r["shared_fields"]:
            print(f"    Data-flow:       A->B shared fields: {r['shared_fields']}")
        print(f"    Match: {' + '.join(markers)}")
        print()

    # Summary
    structural_matches = {n for n, r in results.items() if r["structural"]}
    semantic_matches = {n for n, r in results.items() if r["semantic"]}
    dataflow_matches = {n for n, r in results.items() if r["dataflow"]}

    print("─" * 70)
    print(f"  Structural (topology only):       {sorted(structural_matches)}")
    print(f"  Semantic (+ eval field names):     {sorted(semantic_matches)}")
    print(f"  Data-flow (+ shared fields):       {sorted(dataflow_matches)}")
    print()

    false_positives = structural_matches - semantic_matches
    if false_positives:
        print(f"  False positives fixed by semantics: {sorted(false_positives)}")
    else:
        print("  No false positives to fix (structural and semantic agree)")

    regressions = semantic_matches - structural_matches
    if regressions:
        print(f"  Regressions (semantic-only):        {sorted(regressions)}")

    print(f"\n  Query time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
