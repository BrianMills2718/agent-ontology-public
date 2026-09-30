#!/usr/bin/env python3
"""
Agent Ontology Project Dashboard
==================================
Runs all analysis tools on all specs in a directory and produces
a unified project-health report.

Usage:
    python3 dashboard.py specs/              # full dashboard
    python3 dashboard.py specs/ --json       # JSON output
    python3 dashboard.py specs/ --brief      # just the summary numbers
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median

from .complexity import analyze_spec, load_yaml
from .coverage import analyze_coverage, ENTITY_TYPES, PROCESS_TYPES, EDGE_TYPES, FEATURE_FLAGS, FEATURE_LABELS
from .topology import analyze_topology
from .lint import lint_spec
from .validate import validate_spec

# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _bold(t: str) -> str:
    return _c("1", t)


def _dim(t: str) -> str:
    return _c("2", t)


def _cyan(t: str) -> str:
    return _c("36", t)


# ---------------------------------------------------------------------------
# Spec discovery and agent matching
# ---------------------------------------------------------------------------

def find_specs(directory: str) -> list[str]:
    """Find all .yaml spec files in a directory (non-recursive, sorted)."""
    specs = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith((".yaml", ".yml")):
            specs.append(os.path.join(directory, fname))
    return specs


def _agent_file_for_spec(spec_path: str) -> str | None:
    """Return the expected agent file path for a spec, or None if not found.

    Convention: specs/foo_bar.yaml -> agents/foo_bar_agent.py
    Also handles hyphens: specs/claude-code.yaml -> agents/claude_code_agent.py
    """
    spec_dir = os.path.dirname(os.path.abspath(spec_path))
    project_root = os.path.dirname(spec_dir)
    stem = Path(spec_path).stem  # e.g. "claude-code"
    agent_stem = stem.replace("-", "_")
    agent_path = os.path.join(project_root, "agents", f"{agent_stem}_agent.py")
    if os.path.isfile(agent_path):
        return agent_path
    return None


# ---------------------------------------------------------------------------
# Run all analyses on a single spec
# ---------------------------------------------------------------------------

def analyze_one(spec_path: str, ontology: dict | None = None) -> dict:
    """Run every analysis tool on one spec file. Returns a combined result dict."""
    spec = load_yaml(spec_path)
    fname = os.path.basename(spec_path)
    name = spec.get("name", fname)

    result: dict = {
        "file": fname,
        "name": name,
        "version": spec.get("version", "?"),
    }

    # Agent instantiation check
    agent_path = _agent_file_for_spec(spec_path)
    result["instantiated"] = agent_path is not None
    result["agent_file"] = agent_path

    # Complexity
    complexity = analyze_spec(spec)
    result["complexity"] = complexity

    # Coverage
    coverage = analyze_coverage(spec)
    result["coverage"] = coverage

    # Topology
    topology = analyze_topology(spec)
    result["topology"] = topology

    # Lint
    lint_issues = lint_spec(spec)
    result["lint"] = [
        {"code": i.code, "severity": i.severity, "rule": i.rule_name, "message": i.message}
        for i in lint_issues
    ]

    # Validation
    if ontology is not None:
        errors, warnings = validate_spec(spec, ontology, spec_path)
        result["validation"] = {"errors": errors, "warnings": warnings}
    else:
        result["validation"] = {"errors": [], "warnings": []}

    return result


def analyze_all(spec_dir: str) -> list[dict]:
    """Run all analyses on every spec in a directory."""
    spec_paths = find_specs(spec_dir)
    if not spec_paths:
        return []

    # Load ontology for validation
    project_root = os.path.dirname(os.path.abspath(spec_dir))
    ontology_path = os.path.join(project_root, "ONTOLOGY.yaml")
    ontology = None
    if os.path.isfile(ontology_path):
        ontology = load_yaml(ontology_path)

    results = []
    for sp in spec_paths:
        try:
            results.append(analyze_one(sp, ontology))
        except Exception as exc:
            print(f"  Warning: failed to analyze {os.path.basename(sp)}: {exc}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Report formatting -- full report
# ---------------------------------------------------------------------------

_BAR_WIDTH = 30
_FEATURE_LABELS = FEATURE_LABELS


def _bar(value: float, max_val: float, width: int = _BAR_WIDTH) -> str:
    """Render an ASCII bar chart segment."""
    if max_val <= 0:
        filled = 0
    else:
        filled = int(round(value / max_val * width))
    filled = min(filled, width)
    return "#" * filled + "." * (width - filled)


def _check(val: bool) -> str:
    """Return a checkmark or X mark."""
    return _green("v") if val else _dim("x")


def _visible_len(s: str) -> int:
    """Return the visible length of a string, ignoring ANSI escape codes."""
    import re
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def _pad(s: str, width: int, align: str = "<") -> str:
    """Pad a string to *width* visible characters, respecting ANSI codes."""
    vlen = _visible_len(s)
    pad_needed = max(0, width - vlen)
    if align == ">":
        return " " * pad_needed + s
    elif align == "^":
        left = pad_needed // 2
        right = pad_needed - left
        return " " * left + s + " " * right
    else:  # left-align
        return s + " " * pad_needed


def _score_color(score: float) -> str:
    """Colorize a complexity score."""
    s = f"{score:.0f}"
    if score < 30:
        return _green(s)
    elif score < 60:
        return _yellow(s)
    else:
        return _red(s)


def _severity_color(severity: str, text: str) -> str:
    if severity == "error":
        return _red(text)
    if severity == "warn":
        return _yellow(text)
    return _cyan(text)


def _trunc(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: width - 2] + ".."


def format_full_report(results: list[dict], spec_dir: str) -> str:
    """Build the full dashboard report as a string."""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    project_name = os.path.basename(os.path.dirname(os.path.abspath(spec_dir)))

    # ── 1. Header ─────────────────────────────────────────
    lines.append("")
    lines.append(_bold(f"  Agent Ontology Dashboard: {project_name}"))
    lines.append(f"  {now}  |  {len(results)} specs  |  tools: complexity, coverage, topology, lint, validate")
    lines.append("  " + "=" * 72)

    # ── 2. Test Status ────────────────────────────────────
    lines.append("")
    lines.append(_bold("  SPEC STATUS"))
    lines.append(f"  {'Spec':<22} {'Agent?':>6}  {'Score':>5}  {'Topology':<13} {'Valid':>5}  {'Lint':>4}")
    lines.append("  " + "-" * 64)

    for r in results:
        name = _trunc(r["name"], 20)
        inst = _green("yes") if r["instantiated"] else _red("no ")
        score = _score_color(r["complexity"]["overall_score"])
        topo = r["topology"]["classification"]
        val_err = len(r["validation"]["errors"])
        val_warn = len(r["validation"]["warnings"])
        if val_err > 0:
            valid_str = _red(f"{val_err}E")
        elif val_warn > 0:
            valid_str = _yellow(f"{val_warn}W")
        else:
            valid_str = _green("ok")
        lint_count = len(r["lint"])
        lint_str = str(lint_count) if lint_count == 0 else _yellow(str(lint_count))
        row = (f"  {_pad(name, 22)} {_pad(inst, 6, '>')}  {_pad(score, 5, '>')}  "
               f"{_pad(topo, 13)} {_pad(valid_str, 5, '>')}  {_pad(lint_str, 4, '>')}")
        lines.append(row)

    # ── 3. Complexity Distribution ────────────────────────
    scores = [r["complexity"]["overall_score"] for r in results]
    if scores:
        lines.append("")
        lines.append(_bold("  COMPLEXITY DISTRIBUTION"))
        s_min = min(scores)
        s_max = max(scores)
        s_mean = mean(scores)
        s_median = median(scores)
        lines.append(f"  min={s_min:.0f}  max={s_max:.0f}  mean={s_mean:.1f}  median={s_median:.1f}")
        lines.append("")

        # Build histogram in 5 buckets: 0-20, 20-40, 40-60, 60-80, 80-100
        buckets = [0, 0, 0, 0, 0]
        bucket_labels = ["0-19 ", "20-39", "40-59", "60-79", "80-100"]
        for s in scores:
            idx = min(int(s // 20), 4)
            buckets[idx] += 1
        max_bucket = max(buckets) if buckets else 1
        for i, (label, count) in enumerate(zip(bucket_labels, buckets)):
            bar = _bar(count, max_bucket, 20)
            lines.append(f"  {label} |{bar}| {count}")

    # ── 4. Coverage Summary ───────────────────────────────
    lines.append("")
    lines.append(_bold("  COVERAGE SUMMARY"))

    # Aggregate which ontology types are used across ALL specs
    all_entity_used: set[str] = set()
    all_process_used: set[str] = set()
    all_edge_used: set[str] = set()
    all_feature_used: set[str] = set()

    for r in results:
        cov = r["coverage"]
        all_entity_used.update(cov["entity_types"]["used"])
        all_process_used.update(cov["process_types"]["used"])
        all_edge_used.update(cov["edge_types"]["used"])
        for flag, val in cov["features"]["flags"].items():
            if val:
                all_feature_used.add(flag)

    def _type_line(label: str, used: set, canonical: tuple) -> str:
        used_list = [_green(t) if t in used else _dim(t) for t in canonical]
        count = len(used & set(canonical))
        return f"  {label:16s} {count}/{len(canonical)}  {', '.join(used_list)}"

    lines.append(_type_line("Entity types:", all_entity_used, ENTITY_TYPES))
    lines.append(_type_line("Process types:", all_process_used, PROCESS_TYPES))
    lines.append(_type_line("Edge types:", all_edge_used, EDGE_TYPES))

    feat_parts = [_green(_FEATURE_LABELS[f]) if f in all_feature_used else _dim(_FEATURE_LABELS[f]) for f in FEATURE_FLAGS]
    lines.append(f"  {'Features:':16s} {len(all_feature_used)}/{len(FEATURE_FLAGS)}  {', '.join(feat_parts)}")

    never_used: list[str] = []
    for t in ENTITY_TYPES:
        if t not in all_entity_used:
            never_used.append(t)
    for t in PROCESS_TYPES:
        if t not in all_process_used:
            never_used.append(t)
    for t in EDGE_TYPES:
        if t not in all_edge_used:
            never_used.append(t)
    if never_used:
        lines.append(f"  {_dim('Never used:')}      {', '.join(never_used)}")

    # ── 5. Topology Breakdown ─────────────────────────────
    lines.append("")
    lines.append(_bold("  TOPOLOGY BREAKDOWN"))
    topo_counts: dict[str, int] = defaultdict(int)
    for r in results:
        topo_counts[r["topology"]["classification"]] += 1
    # Sort by count desc
    for topo_name, count in sorted(topo_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        bar = _bar(count, len(results), 15)
        lines.append(f"  {topo_name:<14} {bar} {count:>2} ({pct:.0f}%)")

    # ── 6. Lint Summary ───────────────────────────────────
    lines.append("")
    lines.append(_bold("  LINT SUMMARY"))
    all_lint: list[dict] = []
    for r in results:
        all_lint.extend(r["lint"])

    sev_counts: dict[str, int] = defaultdict(int)
    rule_counts: dict[str, int] = defaultdict(int)
    for issue in all_lint:
        sev_counts[issue["severity"]] += 1
        rule_counts[issue["rule"]] += 1

    total_lint = len(all_lint)
    err_c = sev_counts.get("error", 0)
    warn_c = sev_counts.get("warn", 0)
    info_c = sev_counts.get("info", 0)

    err_str = _red(str(err_c)) if err_c else str(err_c)
    warn_str = _yellow(str(warn_c)) if warn_c else str(warn_c)
    info_str = _cyan(str(info_c)) if info_c else str(info_c)
    lines.append(f"  Total: {total_lint} issues  ({err_str} error, {warn_str} warn, {info_str} info)")

    if rule_counts:
        top_rules = sorted(rule_counts.items(), key=lambda x: -x[1])[:3]
        lines.append(f"  Top violations:")
        for rule, cnt in top_rules:
            lines.append(f"    {rule:<30} {cnt}")

    # ── 7. Feature Matrix ─────────────────────────────────
    lines.append("")
    lines.append(_bold("  FEATURE MATRIX"))

    feature_keys = ["fan_out", "loops", "stores", "tools", "policies", "human_in_loop", "recursive_spawn"]
    feature_hdrs = ["fan-out", "loops", "stores", "tools", "policy", "human", "recurs"]
    col_w = 8

    header_row = f"  {'Spec':<22}" + "".join(f"{h:>{col_w}}" for h in feature_hdrs)
    lines.append(header_row)
    lines.append("  " + "-" * (22 + col_w * len(feature_hdrs)))

    for r in results:
        name = _trunc(r["name"], 20)
        flags = r["coverage"]["features"]["flags"]
        cells = ""
        for fk in feature_keys:
            cells += _pad(_check(flags.get(fk, False)), col_w, ">")
        lines.append(f"  {name:<22}{cells}")

    # ── Footer ────────────────────────────────────────────
    lines.append("")
    lines.append("  " + "=" * 72)
    inst_count = sum(1 for r in results if r["instantiated"])
    avg_coverage = mean([r["coverage"]["overall_pct"] for r in results]) if results else 0
    lines.append(
        f"  {len(results)} specs  |  {inst_count} instantiated  |  "
        f"mean complexity {mean(scores):.0f}  |  mean coverage {avg_coverage:.0f}%  |  "
        f"{total_lint} lint issues"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Brief report
# ---------------------------------------------------------------------------

def format_brief_report(results: list[dict]) -> str:
    """One-screen summary with just the key numbers."""
    lines: list[str] = []
    scores = [r["complexity"]["overall_score"] for r in results]
    coverages = [r["coverage"]["overall_pct"] for r in results]
    total_lint = sum(len(r["lint"]) for r in results)
    total_val_err = sum(len(r["validation"]["errors"]) for r in results)
    inst_count = sum(1 for r in results if r["instantiated"])

    topo_counts: dict[str, int] = defaultdict(int)
    for r in results:
        topo_counts[r["topology"]["classification"]] += 1
    topo_summary = ", ".join(f"{k}:{v}" for k, v in sorted(topo_counts.items(), key=lambda x: -x[1]))

    lines.append("")
    lines.append(_bold("  Agent Ontology Dashboard (brief)"))
    lines.append(f"  Specs:         {len(results)}")
    lines.append(f"  Instantiated:  {inst_count}/{len(results)}")
    lines.append(f"  Complexity:    min={min(scores):.0f}  max={max(scores):.0f}  mean={mean(scores):.1f}  median={median(scores):.1f}")
    lines.append(f"  Coverage:      min={min(coverages):.0f}%  max={max(coverages):.0f}%  mean={mean(coverages):.1f}%")
    lines.append(f"  Topologies:    {topo_summary}")
    lines.append(f"  Lint issues:   {total_lint}")
    lines.append(f"  Validation:    {total_val_err} errors")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def build_json_report(results: list[dict], spec_dir: str) -> dict:
    """Build a JSON-serializable report dict."""
    scores = [r["complexity"]["overall_score"] for r in results]
    coverages = [r["coverage"]["overall_pct"] for r in results]
    total_lint = sum(len(r["lint"]) for r in results)

    topo_counts: dict[str, int] = defaultdict(int)
    for r in results:
        topo_counts[r["topology"]["classification"]] += 1

    sev_counts: dict[str, int] = defaultdict(int)
    rule_counts: dict[str, int] = defaultdict(int)
    for r in results:
        for issue in r["lint"]:
            sev_counts[issue["severity"]] += 1
            rule_counts[issue["rule"]] += 1

    return {
        "generated": datetime.now().isoformat(),
        "spec_dir": spec_dir,
        "spec_count": len(results),
        "instantiated_count": sum(1 for r in results if r["instantiated"]),
        "complexity": {
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
            "mean": round(mean(scores), 1) if scores else 0,
            "median": round(median(scores), 1) if scores else 0,
        },
        "coverage": {
            "min": min(coverages) if coverages else 0,
            "max": max(coverages) if coverages else 0,
            "mean": round(mean(coverages), 1) if coverages else 0,
        },
        "topology_counts": dict(topo_counts),
        "lint": {
            "total": total_lint,
            "by_severity": dict(sev_counts),
            "top_rules": sorted(rule_counts.items(), key=lambda x: -x[1])[:3],
        },
        "specs": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent Ontology Project Dashboard -- unified project health report.",
        epilog="Examples:\n"
               "  python3 dashboard.py specs/              # full dashboard\n"
               "  python3 dashboard.py specs/ --json        # JSON output\n"
               "  python3 dashboard.py specs/ --brief       # summary numbers only\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "spec_dir",
        help="Directory containing spec YAML files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output the full report as JSON.",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Print only summary numbers.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    global _USE_COLOR
    if args.no_color:
        _USE_COLOR = False

    spec_dir = args.spec_dir
    if not os.path.isdir(spec_dir):
        print(f"Error: {spec_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    results = analyze_all(spec_dir)
    if not results:
        print(f"No spec files found in {spec_dir}.", file=sys.stderr)
        sys.exit(1)

    if args.json_output:
        report = build_json_report(results, spec_dir)
        print(json.dumps(report, indent=2, default=str))
    elif args.brief:
        print(format_brief_report(results))
    else:
        print(format_full_report(results, spec_dir))


if __name__ == "__main__":
    main()
