#!/usr/bin/env python3
"""Create a git worktree and fail loud if the fresh checkout is unsafe.

This wrapper exists because raw ``git worktree add`` does not provide any
ecosystem-level guarantee that the created checkout is actually usable for
agent work. A freshly created worktree should be clean. If it is not, that is
already enough to block autonomous execution. The wrapper records the immediate
status, classifies stronger split-brain-like symptoms, and optionally cleans up
the failed worktree instead of leaving an ambiguous checkout in circulation.

When strict coordination enforcement is enabled, the wrapper also requires an
active scoped write claim before the worktree is created.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StatusEntry:
    """One porcelain status entry from a worktree checkout."""

    code: str
    path: str


@dataclass(frozen=True)
class WorktreeStatusSummary:
    """Summarize the immediate status of a freshly created worktree."""

    branch_line: str | None
    entries: list[StatusEntry]
    deleted_count: int
    untracked_count: int
    split_brain_like: bool
    clean: bool


@dataclass(frozen=True)
class WorktreeCreationResult:
    """Structured result for worktree creation and first-status verification."""

    ok: bool
    repo_root: str
    worktree_path: str
    branch: str
    created_branch: bool
    classification: str
    cleanup_performed: bool
    message: str
    status: WorktreeStatusSummary | None
    coordination_checked: bool
    coordination_message: str | None


@dataclass(frozen=True)
class CheckoutStateSummary:
    """Summarize whether one checkout is safe to use as a control surface."""

    clean: bool
    unmerged: bool
    entries: list[StatusEntry]
    modified_count: int
    untracked_count: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for worktree creation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Git repo root")
    parser.add_argument("--path", help="Path for the new worktree")
    parser.add_argument("--branch", help="Branch to create or attach")
    parser.add_argument(
        "--start-point",
        default="HEAD",
        help="Commit-ish to branch from when the branch does not already exist",
    )
    parser.add_argument(
        "--split-brain-threshold",
        type=int,
        default=5,
        help="Deleted/untracked count threshold for split-brain-like classification",
    )
    parser.add_argument(
        "--keep-failed-worktree",
        action="store_true",
        help="Leave a failed worktree on disk for manual diagnosis",
    )
    parser.add_argument(
        "--require-write-claim",
        action="store_true",
        help="Require a matching scoped write claim before creating the worktree.",
    )
    parser.add_argument(
        "--require-clean-main-root",
        action="store_true",
        help="Require the canonical main checkout to be clean before creating the worktree.",
    )
    parser.add_argument("--claim-agent", help="Agent name expected on the scoped write claim.")
    parser.add_argument(
        "--claim-project",
        help="Claim project name. Defaults to the canonical repo root name when omitted.",
    )
    parser.add_argument(
        "--claim-write-path",
        action="append",
        default=[],
        help="Repo-relative write path required by the scoped claim. Repeat as needed.",
    )
    parser.add_argument(
        "--claims-dir",
        help="Override the coordination claims directory instead of ~/.claude/coordination/claims/.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output",
    )
    parser.add_argument(
        "--print-default-worktree-dir",
        action="store_true",
        help="Print the canonical default *_worktrees directory for the repo and exit.",
    )
    return parser.parse_args(argv)


def run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one git command and capture stdout/stderr for diagnosis."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def branch_exists(repo_root: Path, branch: str) -> bool:
    """Return whether a local branch already exists in the target repo."""
    result = run_git(["show-ref", "--verify", f"refs/heads/{branch}"], cwd=repo_root)
    return result.returncode == 0


def resolve_main_repo_root(repo_root: Path) -> Path:
    """Resolve the canonical main repo root from either a root checkout or worktree."""
    result = run_git(
        ["rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repo_root,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Unable to resolve canonical repo root from git common dir:\n"
            f"{result.stderr or result.stdout}".strip()
        )
    git_common_dir = Path(result.stdout.strip())
    return git_common_dir.parent


def get_default_worktree_dir(repo_root: Path) -> Path:
    """Return the canonical repo-level *_worktrees directory for this repo."""
    main_repo_root = resolve_main_repo_root(repo_root.resolve())
    return main_repo_root.parent / f"{main_repo_root.name}_worktrees"


def _load_claims_module() -> Any:
    """Load the sibling coordination-claims script as a module."""
    module_path = Path(__file__).resolve().parents[1] / "check_coordination_claims.py"
    spec = importlib.util.spec_from_file_location("coordination_claims_for_worktree", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load coordination claims module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_status_porcelain(
    porcelain: str,
    *,
    split_brain_threshold: int,
) -> WorktreeStatusSummary:
    """Parse porcelain status output into a deterministic summary."""
    branch_line: str | None = None
    entries: list[StatusEntry] = []
    deleted_count = 0
    untracked_count = 0

    for raw_line in porcelain.splitlines():
        if not raw_line:
            continue
        if raw_line.startswith("## "):
            branch_line = raw_line[3:]
            continue
        if raw_line.startswith("?? "):
            code = "??"
            path = raw_line[3:]
            untracked_count += 1
        else:
            code = raw_line[:2]
            path = raw_line[3:]
            if "D" in code:
                deleted_count += 1
        entries.append(StatusEntry(code=code, path=path))

    split_brain_like = (
        deleted_count >= split_brain_threshold and untracked_count >= split_brain_threshold
    )
    return WorktreeStatusSummary(
        branch_line=branch_line,
        entries=entries,
        deleted_count=deleted_count,
        untracked_count=untracked_count,
        split_brain_like=split_brain_like,
        clean=len(entries) == 0,
    )


def inspect_worktree_state(
    worktree_path: Path,
    *,
    split_brain_threshold: int,
) -> WorktreeStatusSummary:
    """Inspect immediate worktree status after creation."""
    result = run_git(
        ["status", "--porcelain", "--untracked-files=all", "--branch"],
        cwd=worktree_path,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Unable to inspect fresh worktree status:\n"
            f"{result.stderr or result.stdout}".strip()
        )
    return parse_status_porcelain(
        result.stdout,
        split_brain_threshold=split_brain_threshold,
    )


def classify_summary(summary: WorktreeStatusSummary) -> str:
    """Classify the immediate worktree state for operator-facing messages."""
    if summary.clean:
        return "clean"
    if summary.split_brain_like:
        return "split-brain-like"
    return "dirty"


def inspect_checkout_state(checkout_path: Path) -> CheckoutStateSummary:
    """Inspect whether one checkout is safe to use as a control surface."""

    result = run_git(
        ["status", "--porcelain", "--untracked-files=all"],
        cwd=checkout_path,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Unable to inspect checkout status:\n"
            f"{result.stderr or result.stdout}".strip()
        )

    entries: list[StatusEntry] = []
    modified_count = 0
    untracked_count = 0
    unmerged = False

    for raw_line in result.stdout.splitlines():
        if not raw_line:
            continue
        if raw_line.startswith("?? "):
            entries.append(StatusEntry(code="??", path=raw_line[3:]))
            untracked_count += 1
            continue
        code = raw_line[:2]
        path = raw_line[3:]
        entries.append(StatusEntry(code=code, path=path))
        if code in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}:
            unmerged = True
        else:
            modified_count += 1

    return CheckoutStateSummary(
        clean=len(entries) == 0,
        unmerged=unmerged,
        entries=entries,
        modified_count=modified_count,
        untracked_count=untracked_count,
    )


def verify_clean_main_root(repo_root: Path) -> tuple[bool, str]:
    """Require the canonical main checkout to be clean before publish-worktree creation."""

    main_repo_root = resolve_main_repo_root(repo_root)
    summary = inspect_checkout_state(main_repo_root)
    if summary.clean:
        return True, f"Canonical main checkout is clean: {main_repo_root}"

    classification = "main-root-unmerged" if summary.unmerged else "main-root-dirty"
    sample_entries = ", ".join(
        f"{entry.code} {entry.path}" for entry in summary.entries[:8]
    )
    return (
        False,
        "Publish worktree creation blocked: canonical main checkout is not clean. "
        f"classification={classification}; "
        f"path={main_repo_root}; "
        f"modified={summary.modified_count}; "
        f"untracked={summary.untracked_count}; "
        f"sample=[{sample_entries}]. "
        "Do not create a publish lane from a dirty primary checkout; either clear the blocker first or keep the verified branch unpublished on trunk.",
    )


def cleanup_failed_worktree(
    repo_root: Path,
    worktree_path: Path,
    *,
    branch: str,
    created_branch: bool,
) -> tuple[bool, str]:
    """Attempt to remove a failed worktree and delete its just-created branch."""
    remove_result = run_git(["worktree", "remove", "--force", str(worktree_path)], cwd=repo_root)
    if remove_result.returncode != 0 and worktree_path.exists():
        return False, (remove_result.stderr or remove_result.stdout).strip()
    if created_branch:
        delete_result = run_git(["branch", "-D", branch], cwd=repo_root)
        if delete_result.returncode != 0:
            return False, (delete_result.stderr or delete_result.stdout).strip()
    return True, "cleanup complete"


def ensure_safe_target_path(worktree_path: Path) -> None:
    """Reject non-empty target paths before invoking git worktree add."""
    if not worktree_path.exists():
        return
    if worktree_path.is_dir() and not any(worktree_path.iterdir()):
        return
    raise ValueError(
        f"Target worktree path already exists and is not empty: {worktree_path}"
    )


def _claim_project(repo_root: Path, explicit_project: str | None) -> str:
    """Resolve the project name used for scoped claim matching."""
    if explicit_project:
        return explicit_project
    return resolve_main_repo_root(repo_root).name


def _write_paths_are_covered(*, claims_module: Any, required_paths: list[str], claim_paths: list[str]) -> bool:
    """Return whether a scoped claim covers every required write path."""
    for required in required_paths:
        if not any(claims_module._paths_overlap(required, existing) for existing in claim_paths):
            return False
    return True


def verify_scoped_write_claim(
    *,
    repo_root: Path,
    worktree_path: Path,
    branch: str,
    claim_agent: str | None,
    claim_project: str | None,
    claim_write_paths: list[str],
    claims_dir: Path | None,
) -> tuple[bool, str]:
    """Require a matching active write claim and reject conflicting claims."""
    if not claim_agent:
        return False, "Scoped write-claim enforcement requires --claim-agent."
    if not claim_write_paths:
        return False, "Scoped write-claim enforcement requires at least one --claim-write-path."

    claims_module = _load_claims_module()
    if claims_dir is not None:
        claims_module.CLAIMS_DIR = claims_dir.resolve()

    project_name = _claim_project(repo_root, claim_project)
    normalized_paths = [claims_module._normalize_repo_path(path) for path in claim_write_paths]
    candidate = claims_module.build_candidate_claim(
        agent=claim_agent,
        project=project_name,
        scope=f"worktree:{branch}",
        intent=f"Create sanctioned worktree {branch}",
        claim_type="write",
        write_paths=normalized_paths,
        branch=branch,
        worktree_path=str(worktree_path),
    )
    active_claims = claims_module.check_claims(project_name)

    matching_claims = [
        claim
        for claim in active_claims
        if claim.agent == claim_agent
        and claim.claim_type == "write"
        and project_name in claim.projects
        and _write_paths_are_covered(
            claims_module=claims_module,
            required_paths=normalized_paths,
            claim_paths=claim.write_paths,
        )
        and (claim.branch in (None, branch))
    ]
    if not matching_claims:
        joined_paths = ", ".join(normalized_paths)
        return (
            False,
            "Scoped write-claim enforcement failed: no active matching write claim for "
            f"agent={claim_agent}, project={project_name}, branch={branch}, "
            f"write_paths=[{joined_paths}]. Create the narrow write claim first.",
        )

    weak_matching_claims = [
        (claim, claims_module.claim_health_issues(claim))
        for claim in matching_claims
        if claims_module.claim_health_issues(claim)
    ]
    if weak_matching_claims:
        claim, issues = weak_matching_claims[0]
        return (
            False,
            "Scoped write-claim enforcement failed: matching active write claim is weak — "
            f"agent={claim.agent}, project={project_name}, scope={claim.scope}, "
            f"issues=[{', '.join(issues)}]. Refresh the claim with explicit live ownership metadata first.",
        )

    check_result = claims_module.evaluate_claim(candidate, active_claims=active_claims)
    hard_conflicts = check_result.hard_conflicts
    if hard_conflicts:
        formatted = "; ".join(
            f"{item.other_agent} ({item.other_scope}: {', '.join(item.overlapping_write_paths)})"
            for item in hard_conflicts
        )
        return (
            False,
            "Scoped write-claim enforcement failed: conflicting active write claim(s) detected — "
            f"{formatted}",
        )

    matched_scope = matching_claims[0].scope
    return True, f"Scoped write claim verified via {claim_agent}:{project_name}:{matched_scope}."


def create_worktree(
    *,
    repo_root: Path,
    worktree_path: Path,
    branch: str,
    start_point: str,
    split_brain_threshold: int,
    keep_failed_worktree: bool,
    require_write_claim: bool = False,
    claim_agent: str | None = None,
    claim_project: str | None = None,
    claim_write_paths: list[str] | None = None,
    claims_dir: Path | None = None,
    require_clean_main_root: bool = False,
) -> WorktreeCreationResult:
    """Create a worktree, inspect it immediately, and fail loud on unsafe state."""
    repo_root = repo_root.resolve()
    worktree_path = worktree_path.resolve()
    coordination_checked = require_write_claim
    coordination_message: str | None = None

    if require_write_claim:
        claim_ok, coordination_message = verify_scoped_write_claim(
            repo_root=repo_root,
            worktree_path=worktree_path,
            branch=branch,
            claim_agent=claim_agent,
            claim_project=claim_project,
            claim_write_paths=claim_write_paths or [],
            claims_dir=claims_dir,
        )
        if not claim_ok:
            return WorktreeCreationResult(
                ok=False,
                repo_root=str(repo_root),
                worktree_path=str(worktree_path),
                branch=branch,
                created_branch=False,
                classification="coordination-error",
                cleanup_performed=False,
                message=coordination_message,
                status=None,
                coordination_checked=coordination_checked,
                coordination_message=coordination_message,
            )

    if require_clean_main_root:
        clean_main_root, root_message = verify_clean_main_root(repo_root)
        if not clean_main_root:
            return WorktreeCreationResult(
                ok=False,
                repo_root=str(repo_root),
                worktree_path=str(worktree_path),
                branch=branch,
                created_branch=False,
                classification="main-root-dirty",
                cleanup_performed=False,
                message=root_message,
                status=None,
                coordination_checked=coordination_checked,
                coordination_message=coordination_message,
            )

    ensure_safe_target_path(worktree_path)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    branch_already_exists = branch_exists(repo_root, branch)
    if branch_already_exists and start_point != "HEAD":
        raise ValueError(
            "Cannot combine --start-point with an existing branch; "
            f"branch already exists: {branch}"
        )

    add_args = ["worktree", "add", str(worktree_path)]
    if branch_already_exists:
        add_args.append(branch)
        created_branch = False
    else:
        add_args.extend(["-b", branch, start_point])
        created_branch = True

    add_result = run_git(add_args, cwd=repo_root)
    if add_result.returncode != 0:
        return WorktreeCreationResult(
            ok=False,
            repo_root=str(repo_root),
            worktree_path=str(worktree_path),
            branch=branch,
            created_branch=created_branch,
            classification="git-error",
            cleanup_performed=False,
            message=(add_result.stderr or add_result.stdout).strip(),
            status=None,
            coordination_checked=coordination_checked,
            coordination_message=coordination_message,
        )

    summary = inspect_worktree_state(
        worktree_path,
        split_brain_threshold=split_brain_threshold,
    )
    classification = classify_summary(summary)
    if classification == "clean":
        return WorktreeCreationResult(
            ok=True,
            repo_root=str(repo_root),
            worktree_path=str(worktree_path),
            branch=branch,
            created_branch=created_branch,
            classification=classification,
            cleanup_performed=False,
            message="Worktree created cleanly.",
            status=summary,
            coordination_checked=coordination_checked,
            coordination_message=coordination_message,
        )

    cleanup_performed = False
    cleanup_note = ""
    if not keep_failed_worktree:
        cleanup_ok, cleanup_note = cleanup_failed_worktree(
            repo_root,
            worktree_path,
            branch=branch,
            created_branch=created_branch,
        )
        cleanup_performed = cleanup_ok
        if not cleanup_ok and worktree_path.exists():
            cleanup_note = f" Cleanup failed: {cleanup_note}"

    sample_entries = ", ".join(
        f"{entry.code} {entry.path}" for entry in summary.entries[:8]
    )
    return WorktreeCreationResult(
        ok=False,
        repo_root=str(repo_root),
        worktree_path=str(worktree_path),
        branch=branch,
        created_branch=created_branch,
        classification=classification,
        cleanup_performed=cleanup_performed,
        message=(
            "Fresh worktree was not clean immediately after creation. "
            f"classification={classification}; "
            f"deleted={summary.deleted_count}; "
            f"untracked={summary.untracked_count}; "
            f"sample=[{sample_entries}].{cleanup_note}"
        ),
        status=summary,
        coordination_checked=coordination_checked,
        coordination_message=coordination_message,
    )


def _print_human(result: WorktreeCreationResult) -> None:
    """Print a concise operator summary."""
    state = "OK" if result.ok else "FAIL"
    print(f"{state}: {result.message}")
    print(f"repo: {result.repo_root}")
    print(f"path: {result.worktree_path}")
    print(f"branch: {result.branch}")
    if result.coordination_checked and result.coordination_message:
        print(f"coordination: {result.coordination_message}")
    if result.status is not None:
        print(f"classification: {result.classification}")
        print(
            "status-counts: "
            f"deleted={result.status.deleted_count} "
            f"untracked={result.status.untracked_count} "
            f"entries={len(result.status.entries)}"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for safe worktree creation."""
    args = parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    if args.print_default_worktree_dir:
        default_dir = get_default_worktree_dir(repo_root)
        if args.json:
            print(json.dumps({"default_worktree_dir": str(default_dir)}, indent=2))
        else:
            print(default_dir)
        return 0

    if not args.path or not args.branch:
        missing = []
        if not args.path:
            missing.append("--path")
        if not args.branch:
            missing.append("--branch")
        raise SystemExit(f"Missing required arguments for worktree creation: {', '.join(missing)}")

    worktree_path = Path(args.path).expanduser().resolve()
    claims_dir = Path(args.claims_dir).expanduser().resolve() if args.claims_dir else None
    try:
        result = create_worktree(
            repo_root=repo_root,
            worktree_path=worktree_path,
            branch=args.branch,
            start_point=args.start_point,
            split_brain_threshold=args.split_brain_threshold,
            keep_failed_worktree=args.keep_failed_worktree,
            require_write_claim=args.require_write_claim,
            claim_agent=args.claim_agent,
            claim_project=args.claim_project,
            claim_write_paths=args.claim_write_path,
            claims_dir=claims_dir,
            require_clean_main_root=args.require_clean_main_root,
        )
    except (RuntimeError, ValueError) as exc:
        error_result = WorktreeCreationResult(
            ok=False,
            repo_root=str(repo_root),
            worktree_path=str(worktree_path),
            branch=args.branch,
            created_branch=False,
            classification="error",
            cleanup_performed=False,
            message=str(exc),
            status=None,
            coordination_checked=args.require_write_claim,
            coordination_message=None,
        )
        if args.json:
            print(json.dumps(asdict(error_result), indent=2))
        else:
            _print_human(error_result)
        return 1

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        _print_human(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
