"""Generated agents must match what the generator currently produces.

The committed agents were a 2026-02-10 snapshot. `instantiate.py` changed three
times after that, including the whole llm_client migration, and nothing
regenerated: all 67 files differed from the generator's output by ~16,000 lines
while the README told readers to run them.

Two files are deliberately exempt because they carry hand-written behaviour the
generator does not emit. See agents/HAND_EDITED.md.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPECS = REPO_ROOT / "agent_ontology" / "specs"

# Keep in sync with agents/HAND_EDITED.md.
HAND_EDITED = {"babyagi_agent.py", "autogpt_agent.py"}


def _regenerate(backend: str, out: Path) -> None:
    argv = [sys.executable, "-m", "agent_ontology.instantiate",
            "--all", str(SPECS), "-o", str(out)]
    if backend != "custom":
        argv += ["--backend", backend]
    proc = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"generation failed:\n{proc.stderr[-2000:]}"


@pytest.mark.parametrize("backend,committed", [("custom", "agents"), ("langgraph", "agents_lg")])
def test_committed_agents_match_generator(backend: str, committed: str):
    committed_dir = REPO_ROOT / committed
    with tempfile.TemporaryDirectory(prefix=f"ao-regen-{backend}-") as tmp:
        fresh = Path(tmp)
        _regenerate(backend, fresh)

        generated = {p.name for p in fresh.glob("*.py")}
        on_disk = {p.name for p in committed_dir.glob("*.py")}

        missing = sorted(generated - on_disk)
        assert not missing, f"{committed}/ is missing generated agents: {missing}"

        extra = sorted(on_disk - generated - HAND_EDITED)
        assert not extra, f"{committed}/ has agents no spec produces: {extra}"

        drifted = sorted(
            name for name in generated - HAND_EDITED
            if (committed_dir / name).read_text(encoding="utf-8")
            != (fresh / name).read_text(encoding="utf-8")
        )
        assert not drifted, (
            f"{len(drifted)} committed agent(s) differ from the generator's current output. "
            f"Run `ao-instantiate --all agent_ontology/specs/ -o {committed}/"
            + ("" if backend == "custom" else f" --backend {backend}")
            + f"` and restore {sorted(HAND_EDITED)}. Drifted: {drifted}"
        )


def test_hand_edited_list_matches_the_note():
    """The exemption list and the file documenting it cannot diverge."""
    note = (REPO_ROOT / "agents" / "HAND_EDITED.md").read_text(encoding="utf-8")
    for name in HAND_EDITED:
        assert name in note, f"{name} is exempt in the test but undocumented in HAND_EDITED.md"


def test_generation_is_deterministic():
    """The same specs must produce byte-identical code on every run.

    `instantiate.py` built its chained-gate list as a set of strings and
    iterated it directly. Python randomizes string hashing per process, so two
    gate blocks swapped places between runs -- generated code could not be
    diffed, and the drift check above was flaky rather than wrong.
    """
    hashes = []
    for _ in range(2):
        with tempfile.TemporaryDirectory(prefix="ao-determinism-") as tmp:
            out = Path(tmp)
            _regenerate("langgraph", out)
            hashes.append({p.name: p.read_text(encoding="utf-8") for p in sorted(out.glob("*.py"))})
    first, second = hashes
    differing = sorted(n for n in first if first[n] != second.get(n))
    assert not differing, f"generator output is not reproducible for: {differing}"
