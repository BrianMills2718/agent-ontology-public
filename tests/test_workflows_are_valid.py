"""GitHub workflow files must be structurally valid.

This repository executed zero CI jobs between 2026-02-21 and 2026-08-25 because
a workflow file was unparseable. Nothing in the repo could tell you that: the
runs simply failed in 0 seconds, identically, whether the code was fine or on
fire. These checks are cheap and would have caught both invalidating mistakes
made while repairing it.

`yaml.safe_load` is not sufficient. It silently accepts duplicate mapping keys
(last one wins) while GitHub rejects the file outright -- which is exactly how a
second `if:` on one step slipped through during this repair.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).resolve().parent.parent / ".github" / "workflows").glob("*.yml"))


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys."""


def _no_duplicate_keys(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        seen.add(key)
    return yaml.constructor.SafeConstructor.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
)


def test_there_are_workflows():
    assert WORKFLOWS, "no workflow files found"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_has_no_duplicate_keys(path: Path):
    yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_secrets_context_not_used_in_job_level_if(path: Path):
    """`secrets` is unavailable in a job-level `if:` and invalidates the file.

    This is the exact defect that killed CI for six months.
    """
    doc = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    offenders = [
        name for name, job in (doc.get("jobs") or {}).items()
        if "secrets." in str(job.get("if", ""))
    ]
    assert not offenders, (
        f"{path.name}: jobs {offenders} use the `secrets` context in a job-level `if:`. "
        "GitHub rejects the whole file. Put the secret in job-level `env:` and guard the step."
    )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_step_if_does_not_read_its_own_env(path: Path):
    """A step's own `env:` block is not visible to that step's `if:`.

    The smoke-test job guarded both steps on `env.GEMINI_API_KEY` while setting
    that variable in the same steps' `env:`. The condition was always false, so
    the tests never ran and the job reported success.
    """
    doc = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    bad = []
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps", []) or []:
            cond, env = str(step.get("if", "")), step.get("env") or {}
            for var in env:
                if f"env.{var}" in cond:
                    bad.append(f"{job_name}:{step.get('name', '?')} reads env.{var} it sets itself")
    assert not bad, f"{path.name}: {bad}"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_work_steps_are_not_gated_on_an_in_job_capability_check(path: Path):
    """A capability gate belongs on the job, not on the step it guards.

    While repairing CI I added a step that probed for a usable dependency and
    gated the real `run` step on its output. The jobs then reported SUCCESS with
    their only real step skipped -- the same false-green shape as the smoke-test
    job this branch already fixed.

    Job-level `if:` cannot read `secrets` (that is what killed CI in February)
    but it can read `vars`, and a job gated that way shows as SKIPPED, which is
    the truth.

    The rule is narrow on purpose: a `run` step must not be conditional on
    another step's output unless the job itself is gated. An earlier version of
    this test asked whether "every substantive step is conditional", which
    counted `pip install` as substantive and therefore passed against the exact
    commit that had the defect.
    """
    doc = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    bad = []
    for job_name, job in (doc.get("jobs") or {}).items():
        if job.get("if"):
            continue  # gated at the job level: shows as skipped, which is honest
        for step in job.get("steps", []) or []:
            if "run" not in step:
                continue
            cond = str(step.get("if", ""))
            if re.search(r"\bsteps\.[\w-]+\.outputs\b", cond):
                bad.append(f"{job_name}:{step.get('name', '?')}")
    assert not bad, (
        f"{path.name}: {bad} gate real work on an in-job capability check, so the job "
        "reports success having run nothing. Gate the job with `if: vars.X == 'true'` instead."
    )
