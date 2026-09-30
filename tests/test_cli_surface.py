"""The CLI surface must match what pyproject declares and what the README promises.

Two bugs motivated this file, both of which sat undetected for six months:

  * `ao-dashboard` crashed with KeyError('channels') because coverage.py gained
    three feature flags and dashboard.py never gained their labels.
  * `ao-report --all <dir>`, printed verbatim in the README quick start, was not
    a flag the tool accepted.

Neither is subtle. Both survived because nothing ever ran the commands the
README tells a reader to run.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS = REPO_ROOT / "agent_ontology" / "specs"
REACT = SPECS / "react.yaml"


def _console_scripts() -> dict[str, str]:
    """Map console-script name -> module path, from pyproject."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("[project.scripts]", 1)[1]
    entries = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            if line.startswith("[") and entries:
                break
            continue
        match = re.match(r'^([\w-]+)\s*=\s*"([\w.]+):main"$', line)
        if match:
            entries[match.group(1)] = match.group(2)
    return entries


SCRIPTS = _console_scripts()

# Commands that need network, API keys, a long-running server, or a file the
# repo does not carry. Excluded from execution, still covered by the import and
# README-name checks below.
NOT_OFFLINE_RUNNABLE = {
    "ao-benchmark", "ao-evolve", "ao-design", "ao-ingest", "ao-recommend",
    "ao-knowledge-store", "ao-live-viewer", "ao-live-viewer-v2",
    "ao-live-e2e-review", "ao-live-e2e-fixtures", "ao-specgen",
    "ao-import-crewai", "ao-import-autogen", "ao-import-openai-agents",
    "ao-import-google-adk", "ao-trace", "ao-trace-overlay", "ao-migrate",
    "ao-compose", "ao-instantiate", "ao-mutate",
}

# Offline, but gated on extras this repo does not install by default
# (`pip install -e ".[owl,rdf]"`).
REQUIRES_OPTIONAL_DEPS = {
    "ao-ontology-owl", "ao-ontology-rdf", "ao-owl-bridge",
}

_TMP = Path(tempfile.mkdtemp(prefix="ao-cli-smoke-"))


def test_pyproject_declares_scripts():
    assert len(SCRIPTS) > 30, f"expected the full ao-* surface, found {len(SCRIPTS)}"


@pytest.mark.parametrize("name,module", sorted(SCRIPTS.items()))
def test_console_script_module_imports(name: str, module: str):
    """Every declared entry point resolves to a module exposing main()."""
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing and missing != "agent_ontology":
            pytest.skip(f"optional dependency {missing!r} not installed")
        raise
    assert callable(getattr(mod, "main", None)), f"{module} has no callable main()"


# (command name, argv) for everything runnable without keys or network.
OFFLINE_INVOCATIONS = [
    ("ao-validate", [str(REACT)]),
    ("ao-validate", ["--all", str(SPECS)]),
    ("ao-lint", ["--all", str(SPECS), "--severity", "warn"]),
    ("ao-coverage", ["--all", str(SPECS)]),
    ("ao-complexity", ["--all", str(SPECS)]),
    ("ao-topology", ["--all", str(SPECS)]),
    ("ao-similarity", ["--all", str(SPECS), "--top", "3", "--clusters", "3"]),
    ("ao-mermaid", [str(REACT)]),
    ("ao-verify", [str(REACT)]),
    ("ao-dashboard", [str(SPECS)]),
    ("ao-report", ["--all", str(SPECS)]),
    ("ao-spec-diff", [str(REACT), str(SPECS / "autogpt.yaml")]),
    ("ao-patterns", ["--detect", str(REACT)]),
    ("ao-import-langgraph", [
        str(REPO_ROOT / "agents_lg" / "react_agent_lg.py"),
        "-o", str(_TMP / "imported_react.yaml"), "--validate",
    ]),
]


@pytest.mark.parametrize(
    "name,argv", OFFLINE_INVOCATIONS,
    ids=[f"{c[0]}{'-all' if '--all' in c[1] else ''}" for c in OFFLINE_INVOCATIONS],
)
def test_offline_command_runs(name: str, argv: list[str]):
    """Run the command the way the README tells a reader to run it."""
    module = SCRIPTS[name]
    proc = subprocess.run(
        [sys.executable, "-m", module, *argv],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, (
        f"{name} exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )


def test_every_offline_script_is_covered():
    """No runnable command silently escapes the smoke test."""
    covered = {name for name, _ in OFFLINE_INVOCATIONS}
    expected = set(SCRIPTS) - NOT_OFFLINE_RUNNABLE - REQUIRES_OPTIONAL_DEPS
    missing = sorted(expected - covered)
    assert not missing, (
        "these commands are neither smoke-tested nor listed as not-offline-runnable: "
        f"{missing}"
    )


def test_readme_only_names_real_commands():
    """Every ao-* name in the README is an actual console script."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    named = set(re.findall(r"(?<![\w-])(ao-[a-z0-9-]+)", readme))
    unknown = sorted(named - set(SCRIPTS))
    assert not unknown, f"README names commands that do not exist: {unknown}"


# ---------------------------------------------------------------------------
# README figures. These were maintained by hand in six places and were wrong in
# six places: 32 specs claimed against 37 on disk, 254 property tests claimed
# against 294. A count nobody checks is a count that drifts.
# ---------------------------------------------------------------------------

def _readme() -> str:
    return (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def _spec_names() -> set[str]:
    return {p.stem for p in SPECS.glob("*.yaml")}


def test_readme_catalog_lists_every_spec():
    """The Agent Catalog table covers exactly the specs that exist."""
    catalog = _readme().split("## Agent Catalog", 1)[1].split("\n## ", 1)[0]
    listed = set(re.findall(r"^\| `([a-z0-9_-]+)`", catalog, re.M))
    specs = _spec_names()
    assert not (specs - listed), f"specs missing from the catalog: {sorted(specs - listed)}"
    assert not (listed - specs), f"catalog lists specs that do not exist: {sorted(listed - specs)}"


def test_readme_spec_count_is_accurate():
    """Any 'N agent specs' claim in the README matches the corpus."""
    n = len(_spec_names())
    claims = re.findall(r"(\d+) agent specs", _readme())
    assert claims, "README no longer states a spec count; update this test if that is deliberate"
    for claimed in claims:
        assert int(claimed) == n, f"README claims {claimed} agent specs, {n} exist"


def test_readme_property_test_count_is_accurate():
    """Any '(N tests)' claim about test_properties.py matches what it runs."""
    proc = subprocess.run(
        [sys.executable, "tests/test_properties.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    match = re.search(r"Results:\s*(\d+)/(\d+)\s*passed", proc.stdout)
    assert match, f"could not read the property-test summary:\n{proc.stdout[-1000:]}"
    passed, total = int(match.group(1)), int(match.group(2))
    assert passed == total, f"property tests failing: {passed}/{total}"

    claimed = set(re.findall(r"property[- ]based structural tests \((\d+)\)", _readme(), re.I))
    claimed |= set(re.findall(r"property-based tests \(no API keys needed, (\d+) tests\)", _readme(), re.I))
    assert claimed, "README no longer states a property-test count; update this test if deliberate"
    for c in claimed:
        assert int(c) == total, f"README claims {c} property tests, {total} run"
