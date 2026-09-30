#!/usr/bin/env python3
"""
Convert agent runtime call traces into spec-viewer trace-overlay format.

Input:
- trace.json produced by generated agents ({"metrics": ..., "trace": [...]})
- agent spec YAML (for invoke-edge mapping process -> agent)

Output:
- JSON with {"name": "...", "processes": {...}} suitable for spec-viewer Load Trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_trace(trace_data: Any) -> list[dict[str, Any]]:
    if isinstance(trace_data, dict) and isinstance(trace_data.get("trace"), list):
        return trace_data["trace"]
    if isinstance(trace_data, list):
        return trace_data
    raise ValueError("Trace input must be a list or {'trace': [...]} wrapper.")


def _build_agent_label_to_processes(spec: dict[str, Any]) -> dict[str, list[str]]:
    entities = {e.get("id"): e for e in spec.get("entities", [])}
    out: dict[str, list[str]] = {}

    for edge in spec.get("edges", []):
        if edge.get("type") != "invoke":
            continue
        pid = edge.get("from")
        to = edge.get("to")
        ent = entities.get(to, {})
        if ent.get("type") != "agent":
            continue
        label = ent.get("label")
        if not label or not pid:
            continue
        out.setdefault(label, []).append(pid)

    return out


def _match_process_id(agent_label: str, label_to_processes: dict[str, list[str]]) -> str | None:
    if agent_label in label_to_processes:
        return label_to_processes[agent_label][0]

    # Fuzzy fallback for label variants in call traces.
    for known, processes in label_to_processes.items():
        if agent_label.startswith(known) or known in agent_label or agent_label in known:
            return processes[0]
    return None


def convert(trace_data: Any, spec: dict[str, Any]) -> dict[str, Any]:
    trace = _normalize_trace(trace_data)
    label_to_processes = _build_agent_label_to_processes(spec)

    processes: dict[str, dict[str, Any]] = {}

    for call in trace:
        call_agent = str(call.get("agent", "unknown"))
        pid = _match_process_id(call_agent, label_to_processes) or "_unmapped_"
        proc = processes.setdefault(
            pid,
            {
                "exec_count": 0,
                "duration_ms": 0,
                "llm_calls": 0,
            },
        )
        proc["exec_count"] += 1
        proc["llm_calls"] += 1
        proc["duration_ms"] += int(call.get("duration_ms", 0) or 0)

    return {
        "name": f"{spec.get('name', 'Spec')} trace overlay",
        "processes": processes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert call trace to spec-viewer overlay format.")
    parser.add_argument("trace_json", help="Path to trace.json from generated agent")
    parser.add_argument("spec_yaml", help="Path to agent spec YAML")
    parser.add_argument("-o", "--output", default="trace_overlay.json", help="Output overlay JSON path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trace_path = Path(args.trace_json)
    spec_path = Path(args.spec_yaml)
    out_path = Path(args.output)

    trace_data = load_json(trace_path)
    spec = load_yaml(spec_path)
    overlay = convert(trace_data, spec)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(overlay, f, indent=2)

    print(f"Wrote overlay trace: {out_path}")


if __name__ == "__main__":
    main()
