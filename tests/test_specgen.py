#!/usr/bin/env python3
"""
Specgen E2E Test — validates the full description→spec→validate→instantiate pipeline.

Usage:
    python3 test_specgen.py                          # Full E2E (requires API keys)
    python3 test_specgen.py --dry-run                # Check descriptions exist, skip API calls
    python3 test_specgen.py --description panel_review.md  # Test specific description
    python3 test_specgen.py --fix                    # Enable auto-fix retries
    python3 test_specgen.py --json                   # Output results as JSON
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "agent_ontology")
DESCRIPTIONS_DIR = os.path.join(PROJECT_ROOT, "test_descriptions")


def find_descriptions(specific=None):
    """Find all .md files in test_descriptions/."""
    if specific:
        path = os.path.join(DESCRIPTIONS_DIR, specific)
        if not os.path.exists(path):
            path = os.path.join(DESCRIPTIONS_DIR, specific + ".md")
        if os.path.exists(path):
            return [path]
        print(f"Description not found: {specific}")
        return []

    if not os.path.isdir(DESCRIPTIONS_DIR):
        print(f"No test_descriptions/ directory found at {DESCRIPTIONS_DIR}")
        return []

    files = sorted(
        os.path.join(DESCRIPTIONS_DIR, f)
        for f in os.listdir(DESCRIPTIONS_DIR)
        if f.endswith(".md")
    )
    return files


def run_specgen(desc_path, output_path, fix=False, model="gemini-3-flash-preview"):
    """Run specgen.py on a description file. Returns (ok, duration_ms, output)."""
    cmd = [
        sys.executable,
        os.path.join(PACKAGE_DIR, "specgen.py"),
        desc_path,
        "-o", output_path,
        "--validate",
        "--model", model,
    ]
    if fix:
        cmd.append("--fix")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=PACKAGE_DIR, timeout=180)
    duration_ms = int((time.time() - t0) * 1000)

    combined = (result.stdout + result.stderr).strip()
    has_errors = "ERROR" in combined or result.returncode != 0
    return not has_errors, duration_ms, combined


def validate_spec(spec_path):
    """Validate a spec file. Returns (ok, output)."""
    result = subprocess.run(
        [sys.executable, os.path.join(PACKAGE_DIR, "validate.py"), spec_path],
        capture_output=True, text=True, cwd=PACKAGE_DIR
    )
    combined = (result.stdout + result.stderr).strip()
    has_errors = "ERROR" in combined or result.returncode != 0
    return not has_errors, combined


def instantiate_spec(spec_path, agent_path):
    """Instantiate a spec to Python. Returns (ok, output)."""
    result = subprocess.run(
        [sys.executable, os.path.join(PACKAGE_DIR, "instantiate.py"), spec_path, "-o", agent_path],
        capture_output=True, text=True, cwd=PACKAGE_DIR
    )
    combined = (result.stdout + result.stderr).strip()
    return result.returncode == 0, combined


def syntax_check(py_path):
    """Check Python file compiles. Returns (ok, output)."""
    result = subprocess.run(
        [sys.executable, "-c", f"import ast; ast.parse(open('{py_path}').read()); print('OK')"],
        capture_output=True, text=True
    )
    return result.returncode == 0, (result.stdout + result.stderr).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Specgen E2E Test — validate the full description→spec→validate→instantiate pipeline"
    )
    parser.add_argument("--description", "-d", help="Specific description file to test")
    parser.add_argument("--dry-run", action="store_true", help="Check descriptions exist, skip API calls")
    parser.add_argument("--fix", action="store_true", help="Enable auto-fix retries in specgen")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Model to use for specgen")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    descriptions = find_descriptions(args.description)
    if not descriptions:
        print("No descriptions found.")
        sys.exit(1)

    results = []
    work_dir = tempfile.mkdtemp(prefix="specgen_test_")

    if not args.json:
        print(f"\n{'='*60}")
        print(f"  Specgen E2E Test")
        print(f"  {len(descriptions)} description(s), model={args.model}")
        print(f"{'='*60}\n")

    for desc_path in descriptions:
        name = os.path.splitext(os.path.basename(desc_path))[0]
        entry = {
            "name": name,
            "description_path": desc_path,
            "stages": {},
            "status": "pending",
        }

        if args.dry_run:
            entry["status"] = "DRY_RUN_OK"
            entry["stages"]["description"] = {"ok": True, "lines": sum(1 for _ in open(desc_path))}
            results.append(entry)
            if not args.json:
                print(f"  [DRY] {name}: description exists ({entry['stages']['description']['lines']} lines)")
            continue

        if not args.json:
            print(f"  [{name}]")

        # Stage 1: Generate spec
        spec_path = os.path.join(work_dir, f"{name}.yaml")
        ok, dur, output = run_specgen(desc_path, spec_path, fix=args.fix, model=args.model)
        entry["stages"]["specgen"] = {"ok": ok, "duration_ms": dur}
        if not args.json:
            icon = "+" if ok else "-"
            print(f"    {icon} specgen: {'OK' if ok else 'FAIL'} ({dur}ms)")

        if not ok:
            entry["status"] = "SPECGEN_FAIL"
            results.append(entry)
            continue

        # Stage 2: Validate spec
        ok, output = validate_spec(spec_path)
        entry["stages"]["validate"] = {"ok": ok}
        if not args.json:
            icon = "+" if ok else "-"
            print(f"    {icon} validate: {'OK' if ok else 'FAIL'}")

        if not ok:
            entry["status"] = "VALIDATE_FAIL"
            results.append(entry)
            continue

        # Stage 3: Instantiate
        agent_path = os.path.join(work_dir, f"{name}_agent.py")
        ok, output = instantiate_spec(spec_path, agent_path)
        entry["stages"]["instantiate"] = {"ok": ok}
        if not args.json:
            icon = "+" if ok else "-"
            print(f"    {icon} instantiate: {'OK' if ok else 'FAIL'}")

        if not ok:
            entry["status"] = "INSTANTIATE_FAIL"
            results.append(entry)
            continue

        # Stage 4: Syntax check
        ok, output = syntax_check(agent_path)
        entry["stages"]["syntax"] = {"ok": ok}
        if not args.json:
            icon = "+" if ok else "-"
            print(f"    {icon} syntax: {'OK' if ok else 'FAIL'}")

        if not ok:
            entry["status"] = "SYNTAX_FAIL"
            results.append(entry)
            continue

        entry["status"] = "PASS"
        results.append(entry)

    # Cleanup
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        passed = sum(1 for r in results if r["status"] in ("PASS", "DRY_RUN_OK"))
        total = len(results)
        print(f"\n{'='*60}")
        print(f"  Results: {passed}/{total} passed")
        for r in results:
            icon = "+" if r["status"] in ("PASS", "DRY_RUN_OK") else "-"
            print(f"    {icon} {r['name']}: {r['status']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
