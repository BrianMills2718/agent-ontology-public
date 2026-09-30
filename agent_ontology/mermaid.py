#!/usr/bin/env python3
"""
Agent Ontology Spec to Mermaid Converter
==========================================
Converts YAML agent specs to Mermaid flowchart diagrams.

Usage:
    python3 mermaid.py specs/react.yaml                    # Print mermaid to stdout
    python3 mermaid.py specs/react.yaml -o react.mmd       # Write to file
    python3 mermaid.py --all specs/                         # Print all specs
    python3 mermaid.py specs/react.yaml --direction LR      # Left-to-right layout
    python3 mermaid.py specs/react.yaml --no-entities       # Only show process flow
"""

import argparse
import os
import re
import sys

import yaml


# -- YAML loading ------------------------------------------------------------

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# -- Mermaid escaping --------------------------------------------------------

def _escape_label(text):
    """Escape characters that conflict with Mermaid syntax.

    Mermaid uses several bracket/brace styles to denote node shapes, so
    labels containing those characters must be sanitized.  Double-quotes
    inside labels are replaced with single-quotes.  We strip characters
    that could break node definitions: ( ) [ ] { } < > / | #
    """
    if not text:
        return ""
    text = str(text)
    # Replace double-quotes with single-quotes
    text = text.replace('"', "'")
    # Remove characters that interfere with Mermaid node delimiters
    for ch in "()[]{}|<>/#":
        text = text.replace(ch, "")
    # Collapse any resulting multi-spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _escape_edge_label(text):
    """Escape an edge label for Mermaid pipe-delimited labels (|...|)."""
    if not text:
        return ""
    text = str(text)
    text = text.replace('"', "'")
    text = text.replace("|", "/")
    # Remove characters that break Mermaid edge labels
    for ch in "[]{}#":
        text = text.replace(ch, "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -- Node shape templates ----------------------------------------------------

# Each function returns a Mermaid node definition string: "id<shape>label</shape>"
NODE_SHAPES = {
    # Entities
    "agent":        lambda nid, label: f'{nid}[/"{label}"/]',
    "store":        lambda nid, label: f'{nid}[("{label}")]',
    "tool":         lambda nid, label: f"{nid}" + '{{' + f'"{label}"' + '}}',
    "human":        lambda nid, label: f'{nid}("{label}")',
    "config":       lambda nid, label: f'{nid}["{label}"]',
    "channel":      lambda nid, label: f'{nid}[/"{label}"\\]',  # parallelogram
    "team":         lambda nid, label: f'{nid}(("{label}"))',
    "conversation": lambda nid, label: f'{nid}("{label}")',
    # Processes
    "step":          lambda nid, label: f'{nid}["{label}"]',
    "gate":          lambda nid, label: f"{nid}" + '{' + f'"{label}"' + '}',
    "checkpoint":    lambda nid, label: f'{nid}[["{label}"]]',
    "spawn":         lambda nid, label: f'{nid}(("{label}"))',
    "protocol":      lambda nid, label: f'{nid}("{label}")',
    "policy":        lambda nid, label: f'{nid}["{label}"]',
    "error_handler": lambda nid, label: f'{nid}["{label}"]',
}

ENTITY_TYPES = {"agent", "store", "tool", "human", "config", "channel", "team", "conversation"}
PROCESS_TYPES = {"step", "gate", "checkpoint", "spawn", "protocol", "policy", "error_handler"}


# -- Edge rendering ----------------------------------------------------------

def _render_edge(edge, id_to_type):
    """Return a single Mermaid edge line for the given edge dict."""
    etype = edge.get("type", "flow")
    src = edge.get("from", "")
    dst = edge.get("to", "")
    condition = edge.get("condition", "")
    label = edge.get("label", "")

    if etype == "flow":
        if label:
            return f'    {src} -->|"{_escape_edge_label(label)}"| {dst}'
        return f"    {src} --> {dst}"

    elif etype == "invoke":
        elabel = _escape_edge_label(label) if label else "invoke"
        return f'    {src} -.->|"{elabel}"| {dst}'

    elif etype == "branch":
        elabel = _escape_edge_label(condition) if condition else _escape_edge_label(label)
        if elabel:
            return f'    {src} -->|"{elabel}"| {dst}'
        return f"    {src} --> {dst}"

    elif etype == "loop":
        elabel = _escape_edge_label(condition) if condition else _escape_edge_label(label)
        if elabel:
            return f'    {src} -.->|"loop: {elabel}"| {dst}'
        return f"    {src} -.->|loop| {dst}"

    elif etype == "read":
        elabel = _escape_edge_label(label) if label else "read"
        return f'    {src} -.->|"{elabel}"| {dst}'

    elif etype == "write":
        elabel = _escape_edge_label(label) if label else "write"
        return f'    {src} -.->|"{elabel}"| {dst}'

    elif etype == "modify":
        elabel = _escape_edge_label(label) if label else "modify"
        return f'    {src} -.->|"{elabel}"| {dst}'

    elif etype == "observe":
        elabel = _escape_edge_label(label) if label else "observe"
        return f'    {src} -.->|"{elabel}"| {dst}'

    elif etype == "error":
        elabel = _escape_edge_label(label) if label else "error"
        return f'    {src} -.->|"{elabel}"| {dst}'

    elif etype == "publish":
        elabel = _escape_edge_label(label) if label else "publish"
        return f'    {src} -->|"{elabel}"| {dst}'

    elif etype == "subscribe":
        elabel = _escape_edge_label(label) if label else "subscribe"
        return f'    {src} -->|"{elabel}"| {dst}'

    elif etype == "handoff":
        elabel = _escape_edge_label(label) if label else "handoff"
        return f'    {src} ==>|"{elabel}"| {dst}'

    else:
        # Fallback for unknown edge types: dotted with type label
        elabel = _escape_edge_label(label) if label else etype
        return f'    {src} -.->|"{elabel}"| {dst}'


# -- Main conversion ---------------------------------------------------------

def spec_to_mermaid(spec, direction="TD", include_entities=True):
    """Convert a loaded spec dict to a Mermaid flowchart string.

    Args:
        spec: Parsed YAML spec dict.
        direction: Flowchart direction (TD, LR, BT, RL).
        include_entities: If False, entity nodes and their edges are omitted.

    Returns:
        A string containing a complete Mermaid flowchart definition.
    """
    lines = [f"flowchart {direction}"]

    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])

    # Build lookup: id -> type
    id_to_type = {}
    for ent in entities:
        id_to_type[ent.get("id", "")] = ent.get("type", "agent")
    for proc in processes:
        id_to_type[proc.get("id", "")] = proc.get("type", "step")

    # Track entity ids for filtering
    entity_ids = {ent.get("id", "") for ent in entities}

    # -- Entity nodes --
    if include_entities and entities:
        lines.append("")
        lines.append("    %% Entities")
        for ent in entities:
            nid = ent.get("id", "")
            ntype = ent.get("type", "agent")
            label = _escape_label(ent.get("label", nid))
            shape_fn = NODE_SHAPES.get(ntype)
            if shape_fn:
                lines.append(f"    {shape_fn(nid, label)}")
            else:
                lines.append(f'    {nid}["{label}"]')

    # -- Process nodes --
    if processes:
        lines.append("")
        lines.append("    %% Processes")
        for proc in processes:
            nid = proc.get("id", "")
            ntype = proc.get("type", "step")
            label = _escape_label(proc.get("label", nid))
            shape_fn = NODE_SHAPES.get(ntype)
            if shape_fn:
                lines.append(f"    {shape_fn(nid, label)}")
            else:
                lines.append(f'    {nid}["{label}"]')

    # -- Edges --
    if edges:
        lines.append("")
        lines.append("    %% Edges")
        for edge in edges:
            src = edge.get("from", "")
            dst = edge.get("to", "")

            # Skip entity-related edges when --no-entities
            if not include_entities:
                if src in entity_ids or dst in entity_ids:
                    continue

            line = _render_edge(edge, id_to_type)
            if line:
                lines.append(line)

    lines.append("")
    return "\n".join(lines)


# -- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Agent Ontology Spec to Mermaid Converter -- generate Mermaid flowcharts from agent specs",
        epilog="Examples:\n"
               "  python3 mermaid.py specs/react.yaml\n"
               "  python3 mermaid.py specs/react.yaml -o react.mmd\n"
               "  python3 mermaid.py --all specs/\n"
               "  python3 mermaid.py specs/react.yaml --direction LR\n"
               "  python3 mermaid.py specs/react.yaml --no-entities\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("spec", help="Path to a spec YAML file, or directory with --all")
    parser.add_argument("-o", "--output", default=None,
                        help="Write output to a file instead of stdout")
    parser.add_argument("--all", action="store_true",
                        help="Convert all *.yaml files in the given directory")
    parser.add_argument("--direction", default="TD", choices=["TD", "LR", "BT", "RL"],
                        help="Flowchart direction (default: TD)")
    parser.add_argument("--no-entities", action="store_true",
                        help="Omit entity nodes and their edges (show only process flow)")

    args = parser.parse_args()

    include_entities = not args.no_entities

    if args.all:
        # -- Multi-spec mode --
        spec_dir = args.spec
        if not os.path.isdir(spec_dir):
            print(f"Error: {spec_dir} is not a directory (use --all with a directory)", file=sys.stderr)
            sys.exit(1)

        results = []
        for fname in sorted(os.listdir(spec_dir)):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(spec_dir, fname)
            try:
                spec = load_yaml(path)
                name = spec.get("name", fname)
                mermaid = spec_to_mermaid(spec, direction=args.direction,
                                          include_entities=include_entities)
                results.append((name, fname, mermaid))
            except Exception as exc:
                print(f"Warning: failed to convert {fname}: {exc}", file=sys.stderr)

        if not results:
            print("No spec files found.", file=sys.stderr)
            sys.exit(1)

        output_parts = []
        for name, fname, mermaid in results:
            output_parts.append(f"## {name} ({fname})\n")
            output_parts.append(f"```mermaid\n{mermaid}```\n")

        output = "\n".join(output_parts)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Wrote {len(results)} diagrams to {args.output}", file=sys.stderr)
        else:
            print(output)

    else:
        # -- Single-spec mode --
        if not os.path.isfile(args.spec):
            print(f"Error: {args.spec} not found", file=sys.stderr)
            sys.exit(1)

        spec = load_yaml(args.spec)
        mermaid = spec_to_mermaid(spec, direction=args.direction,
                                  include_entities=include_entities)

        if args.output:
            with open(args.output, "w") as f:
                f.write(mermaid + "\n")
            name = spec.get("name", os.path.basename(args.spec))
            print(f"Wrote {name} diagram to {args.output}", file=sys.stderr)
        else:
            print(mermaid)


if __name__ == "__main__":
    main()
