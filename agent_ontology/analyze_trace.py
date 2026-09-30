#!/usr/bin/env python3
"""
Trace Analyzer for agent_ontology trace.json files.

Reads trace.json files produced by agent runs and generates structured
analysis reports covering summary statistics, per-agent breakdowns,
timelines, token estimation, schema compliance, and data flow.

Usage:
    python3 analyze_trace.py trace.json
    python3 analyze_trace.py trace.json --json
    python3 analyze_trace.py --compare trace1.json trace2.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trace(path: str) -> list[dict]:
    """Load and validate a trace.json file. Returns list of call records.

    Accepts both raw JSON arrays and the wrapped format
    ``{"metrics": {...}, "trace": [...]}``.
    """
    p = Path(path)
    if not p.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Handle wrapped format: {"metrics": ..., "trace": [...]}
    if isinstance(data, dict) and "trace" in data:
        data = data["trace"]
    if not isinstance(data, list):
        print(f"Error: expected a JSON array (or {{\"trace\": [...]}} dict) in {path}", file=sys.stderr)
        sys.exit(1)
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


def safe_parse_json(text: str) -> tuple[bool, Any]:
    """Try to parse text as JSON. Returns (success, parsed_or_None)."""
    try:
        return True, json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, None


def format_ms(ms: float) -> str:
    """Format milliseconds into a human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60_000:
        return f"{ms / 1000:.2f}s"
    else:
        minutes = int(ms // 60_000)
        seconds = (ms % 60_000) / 1000
        return f"{minutes}m {seconds:.1f}s"


def extract_json_keys(text: str) -> set[str]:
    """Parse text as JSON and return top-level keys if it is a dict."""
    ok, parsed = safe_parse_json(text)
    if ok and isinstance(parsed, dict):
        return set(parsed.keys())
    return set()


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_summary(trace: list[dict]) -> dict:
    """Compute high-level summary statistics for a trace."""
    total_calls = len(trace)
    durations = [c.get("duration_ms", 0) for c in trace]
    total_duration = sum(durations)
    agents = sorted({c.get("agent", "unknown") for c in trace})
    models = sorted({c.get("model", "unknown") for c in trace})

    total_input_chars = 0
    total_output_chars = 0
    for c in trace:
        total_input_chars += len(c.get("system_prompt", "")) + len(c.get("user_message", ""))
        total_output_chars += len(c.get("response", ""))

    return {
        "total_calls": total_calls,
        "total_duration_ms": total_duration,
        "avg_duration_ms": round(total_duration / total_calls, 1) if total_calls else 0,
        "unique_agents": agents,
        "unique_models": models,
        "estimated_input_tokens": estimate_tokens("x" * total_input_chars),
        "estimated_output_tokens": estimate_tokens("x" * total_output_chars),
    }


def compute_per_agent(trace: list[dict]) -> dict[str, dict]:
    """Compute per-agent breakdown statistics."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for c in trace:
        buckets[c.get("agent", "unknown")].append(c)

    result = {}
    for agent, calls in sorted(buckets.items()):
        durations = [c.get("duration_ms", 0) for c in calls]
        input_lengths = [
            len(c.get("system_prompt", "")) + len(c.get("user_message", ""))
            for c in calls
        ]
        output_lengths = [len(c.get("response", "")) for c in calls]
        result[agent] = {
            "call_count": len(calls),
            "total_duration_ms": sum(durations),
            "avg_duration_ms": round(sum(durations) / len(durations), 1),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "avg_input_chars": round(sum(input_lengths) / len(input_lengths), 1),
            "avg_output_chars": round(sum(output_lengths) / len(output_lengths), 1),
            "avg_input_tokens_est": round(sum(input_lengths) / len(input_lengths) / 4, 1),
            "avg_output_tokens_est": round(sum(output_lengths) / len(output_lengths) / 4, 1),
            "models_used": sorted({c.get("model", "unknown") for c in calls}),
        }
    return result


def compute_timeline(trace: list[dict]) -> list[dict]:
    """Produce a timeline of calls in chronological order."""
    entries = []
    for i, c in enumerate(trace):
        entries.append({
            "index": i,
            "timestamp": c.get("timestamp", ""),
            "agent": c.get("agent", "unknown"),
            "model": c.get("model", "unknown"),
            "duration_ms": c.get("duration_ms", 0),
            "input_chars": len(c.get("system_prompt", "")) + len(c.get("user_message", "")),
            "output_chars": len(c.get("response", "")),
        })
    # Sort by timestamp (they should already be in order, but be safe)
    entries.sort(key=lambda e: e["timestamp"])
    return entries


def compute_token_estimates(trace: list[dict]) -> dict:
    """Compute per-call and aggregate token estimates."""
    per_call = []
    total_input = 0
    total_output = 0
    for i, c in enumerate(trace):
        input_chars = len(c.get("system_prompt", "")) + len(c.get("user_message", ""))
        output_chars = len(c.get("response", ""))
        input_tokens = estimate_tokens("x" * input_chars)
        output_tokens = estimate_tokens("x" * output_chars)
        total_input += input_tokens
        total_output += output_tokens
        per_call.append({
            "index": i,
            "agent": c.get("agent", "unknown"),
            "input_tokens_est": input_tokens,
            "output_tokens_est": output_tokens,
        })
    return {
        "total_input_tokens_est": total_input,
        "total_output_tokens_est": total_output,
        "total_tokens_est": total_input + total_output,
        "per_call": per_call,
    }


def compute_schema_compliance(trace: list[dict]) -> dict:
    """Check how many responses successfully parse as JSON."""
    total = len(trace)
    successes = 0
    failures = []
    for i, c in enumerate(trace):
        ok, _ = safe_parse_json(c.get("response", ""))
        if ok:
            successes += 1
        else:
            failures.append({
                "index": i,
                "agent": c.get("agent", "unknown"),
                "response_preview": c.get("response", "")[:120],
            })
    return {
        "total": total,
        "json_parse_success": successes,
        "json_parse_failure": total - successes,
        "success_rate": round(successes / total * 100, 1) if total else 0.0,
        "failures": failures,
    }


def compute_data_flow(trace: list[dict]) -> list[dict]:
    """Analyze JSON inputs/outputs to show what fields each agent received and produced."""
    flow = []
    for i, c in enumerate(trace):
        input_keys = extract_json_keys(c.get("user_message", ""))
        output_keys = extract_json_keys(c.get("response", ""))
        flow.append({
            "index": i,
            "agent": c.get("agent", "unknown"),
            "input_fields": sorted(input_keys),
            "output_fields": sorted(output_keys),
        })
    return flow


def compute_data_flow_summary(trace: list[dict]) -> dict[str, dict]:
    """Aggregate data flow per agent: union of all input/output fields observed."""
    agent_inputs: dict[str, set[str]] = defaultdict(set)
    agent_outputs: dict[str, set[str]] = defaultdict(set)
    for c in trace:
        agent = c.get("agent", "unknown")
        agent_inputs[agent] |= extract_json_keys(c.get("user_message", ""))
        agent_outputs[agent] |= extract_json_keys(c.get("response", ""))
    all_agents = sorted(set(agent_inputs.keys()) | set(agent_outputs.keys()))
    return {
        agent: {
            "input_fields": sorted(agent_inputs.get(agent, set())),
            "output_fields": sorted(agent_outputs.get(agent, set())),
        }
        for agent in all_agents
    }


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

def analyze_trace(trace: list[dict]) -> dict:
    """Run all analyses on a trace and return a structured report."""
    return {
        "summary": compute_summary(trace),
        "per_agent": compute_per_agent(trace),
        "timeline": compute_timeline(trace),
        "token_estimates": compute_token_estimates(trace),
        "schema_compliance": compute_schema_compliance(trace),
        "data_flow": compute_data_flow(trace),
        "data_flow_summary": compute_data_flow_summary(trace),
    }


# ---------------------------------------------------------------------------
# Human-readable formatting
# ---------------------------------------------------------------------------

def print_divider(title: str, width: int = 72) -> None:
    padding = width - len(title) - 4
    left = padding // 2
    right = padding - left
    print(f"\n{'=' * left}  {title}  {'=' * right}")


def render_summary(summary: dict) -> None:
    print_divider("SUMMARY")
    print(f"  Total LLM calls:          {summary['total_calls']}")
    print(f"  Total duration:            {format_ms(summary['total_duration_ms'])}")
    print(f"  Avg duration per call:     {format_ms(summary['avg_duration_ms'])}")
    print(f"  Unique agents:             {', '.join(summary['unique_agents'])}")
    print(f"  Unique models:             {', '.join(summary['unique_models'])}")
    print(f"  Est. total input tokens:   {summary['estimated_input_tokens']:,}")
    print(f"  Est. total output tokens:  {summary['estimated_output_tokens']:,}")


def render_per_agent(per_agent: dict) -> None:
    print_divider("PER-AGENT BREAKDOWN")
    for agent, stats in per_agent.items():
        print(f"\n  [{agent}]")
        print(f"    Calls:            {stats['call_count']}")
        print(f"    Models:           {', '.join(stats['models_used'])}")
        print(f"    Duration total:   {format_ms(stats['total_duration_ms'])}")
        print(f"    Duration avg:     {format_ms(stats['avg_duration_ms'])}")
        print(f"    Duration min:     {format_ms(stats['min_duration_ms'])}")
        print(f"    Duration max:     {format_ms(stats['max_duration_ms'])}")
        print(f"    Avg input chars:  {stats['avg_input_chars']:.0f}  (~{stats['avg_input_tokens_est']:.0f} tokens)")
        print(f"    Avg output chars: {stats['avg_output_chars']:.0f}  (~{stats['avg_output_tokens_est']:.0f} tokens)")


def render_timeline(timeline: list[dict]) -> None:
    print_divider("TIMELINE")
    # Header
    print(f"  {'#':<4} {'Timestamp':<28} {'Agent':<24} {'Model':<28} {'Duration':>10}")
    print(f"  {'─' * 4} {'─' * 28} {'─' * 24} {'─' * 28} {'─' * 10}")
    for entry in timeline:
        idx = entry["index"]
        ts = entry["timestamp"]
        agent = entry["agent"]
        model = entry["model"]
        dur = format_ms(entry["duration_ms"])
        print(f"  {idx:<4} {ts:<28} {agent:<24} {model:<28} {dur:>10}")


def render_token_estimates(token_est: dict) -> None:
    print_divider("TOKEN ESTIMATES")
    print(f"  Total input tokens (est):  {token_est['total_input_tokens_est']:,}")
    print(f"  Total output tokens (est): {token_est['total_output_tokens_est']:,}")
    print(f"  Total tokens (est):        {token_est['total_tokens_est']:,}")
    print()
    print(f"  {'#':<4} {'Agent':<24} {'Input tok':>12} {'Output tok':>12}")
    print(f"  {'─' * 4} {'─' * 24} {'─' * 12} {'─' * 12}")
    for pc in token_est["per_call"]:
        print(f"  {pc['index']:<4} {pc['agent']:<24} {pc['input_tokens_est']:>12,} {pc['output_tokens_est']:>12,}")


def render_schema_compliance(compliance: dict) -> None:
    print_divider("SCHEMA COMPLIANCE")
    print(f"  Total responses:     {compliance['total']}")
    print(f"  JSON parse success:  {compliance['json_parse_success']}")
    print(f"  JSON parse failure:  {compliance['json_parse_failure']}")
    print(f"  Success rate:        {compliance['success_rate']}%")
    if compliance["failures"]:
        print("\n  Failed responses:")
        for f in compliance["failures"]:
            preview = f["response_preview"].replace("\n", "\\n")
            print(f"    Call #{f['index']} ({f['agent']}): {preview}...")


def render_data_flow(flow_summary: dict) -> None:
    print_divider("DATA FLOW (per agent)")
    for agent, info in flow_summary.items():
        print(f"\n  [{agent}]")
        if info["input_fields"]:
            print(f"    Receives: {', '.join(info['input_fields'])}")
        else:
            print(f"    Receives: (non-JSON or no input)")
        if info["output_fields"]:
            print(f"    Produces: {', '.join(info['output_fields'])}")
        else:
            print(f"    Produces: (non-JSON or no output)")


def render_report(report: dict) -> None:
    """Render the full human-readable report to stdout."""
    render_summary(report["summary"])
    render_per_agent(report["per_agent"])
    render_timeline(report["timeline"])
    render_token_estimates(report["token_estimates"])
    render_schema_compliance(report["schema_compliance"])
    render_data_flow(report["data_flow_summary"])
    print()


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def compare_traces(path_a: str, path_b: str, as_json: bool = False) -> None:
    """Load two traces and print a side-by-side comparison."""
    trace_a = load_trace(path_a)
    trace_b = load_trace(path_b)
    report_a = analyze_trace(trace_a)
    report_b = analyze_trace(trace_b)

    if as_json:
        output = {
            "trace_a": {"file": path_a, "report": report_a},
            "trace_b": {"file": path_b, "report": report_b},
            "delta": build_comparison_delta(report_a, report_b),
        }
        print(json.dumps(output, indent=2, default=str))
        return

    name_a = Path(path_a).name
    name_b = Path(path_b).name

    print_divider("COMPARISON")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")

    sa = report_a["summary"]
    sb = report_b["summary"]

    print_divider("SUMMARY COMPARISON")
    rows = [
        ("Total calls", sa["total_calls"], sb["total_calls"], ""),
        ("Total duration", sa["total_duration_ms"], sb["total_duration_ms"], "ms"),
        ("Avg duration", sa["avg_duration_ms"], sb["avg_duration_ms"], "ms"),
        ("Est. input tokens", sa["estimated_input_tokens"], sb["estimated_input_tokens"], ""),
        ("Est. output tokens", sa["estimated_output_tokens"], sb["estimated_output_tokens"], ""),
    ]
    print(f"  {'Metric':<28} {name_a:>16} {name_b:>16} {'Delta':>16}")
    print(f"  {'─' * 28} {'─' * 16} {'─' * 16} {'─' * 16}")
    for label, va, vb, suffix in rows:
        delta = vb - va
        sign = "+" if delta > 0 else ""
        if suffix == "ms":
            print(f"  {label:<28} {format_ms(va):>16} {format_ms(vb):>16} {sign + format_ms(abs(delta)):>16}")
        else:
            print(f"  {label:<28} {va:>16,} {vb:>16,} {sign}{delta:>15,}")

    # Agents comparison
    all_agents = sorted(set(list(report_a["per_agent"].keys()) + list(report_b["per_agent"].keys())))
    print_divider("PER-AGENT COMPARISON")
    for agent in all_agents:
        stats_a = report_a["per_agent"].get(agent)
        stats_b = report_b["per_agent"].get(agent)
        print(f"\n  [{agent}]")
        print(f"    {'Metric':<24} {name_a:>14} {name_b:>14} {'Delta':>14}")
        print(f"    {'─' * 24} {'─' * 14} {'─' * 14} {'─' * 14}")

        agent_rows = [
            ("Calls", "call_count"),
            ("Total duration (ms)", "total_duration_ms"),
            ("Avg duration (ms)", "avg_duration_ms"),
            ("Avg input chars", "avg_input_chars"),
            ("Avg output chars", "avg_output_chars"),
        ]
        for label, key in agent_rows:
            va = stats_a[key] if stats_a else 0
            vb = stats_b[key] if stats_b else 0
            delta = vb - va
            sign = "+" if delta > 0 else ""
            print(f"    {label:<24} {va:>14.1f} {vb:>14.1f} {sign}{delta:>13.1f}")

    # Schema compliance comparison
    ca = report_a["schema_compliance"]
    cb = report_b["schema_compliance"]
    print_divider("SCHEMA COMPLIANCE COMPARISON")
    print(f"  {'Metric':<28} {name_a:>16} {name_b:>16}")
    print(f"  {'─' * 28} {'─' * 16} {'─' * 16}")
    print(f"  {'Success rate':<28} {ca['success_rate']:>15.1f}% {cb['success_rate']:>15.1f}%")
    print(f"  {'Failures':<28} {ca['json_parse_failure']:>16} {cb['json_parse_failure']:>16}")

    print()


def build_comparison_delta(report_a: dict, report_b: dict) -> dict:
    """Build a delta dict for JSON comparison output."""
    sa = report_a["summary"]
    sb = report_b["summary"]
    return {
        "total_calls": sb["total_calls"] - sa["total_calls"],
        "total_duration_ms": sb["total_duration_ms"] - sa["total_duration_ms"],
        "avg_duration_ms": round(sb["avg_duration_ms"] - sa["avg_duration_ms"], 1),
        "estimated_input_tokens": sb["estimated_input_tokens"] - sa["estimated_input_tokens"],
        "estimated_output_tokens": sb["estimated_output_tokens"] - sa["estimated_output_tokens"],
        "schema_success_rate_a": report_a["schema_compliance"]["success_rate"],
        "schema_success_rate_b": report_b["schema_compliance"]["success_rate"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze agent_ontology trace.json files.",
        epilog=(
            "Examples:\n"
            "  python3 analyze_trace.py trace.json\n"
            "  python3 analyze_trace.py trace.json --json\n"
            "  python3 analyze_trace.py --compare run1.json run2.json\n"
            "  python3 analyze_trace.py --compare run1.json run2.json --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "trace_file",
        nargs="?",
        help="Path to a trace.json file to analyze",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output the report as JSON instead of human-readable text",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("TRACE_A", "TRACE_B"),
        help="Compare two trace files side by side",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        compare_traces(args.compare[0], args.compare[1], as_json=args.json_output)
        return

    # Single-file analysis mode
    if not args.trace_file:
        parser.print_help()
        sys.exit(1)

    trace = load_trace(args.trace_file)
    report = analyze_trace(trace)

    if args.json_output:
        print(json.dumps(report, indent=2, default=str))
    else:
        render_report(report)


if __name__ == "__main__":
    main()
