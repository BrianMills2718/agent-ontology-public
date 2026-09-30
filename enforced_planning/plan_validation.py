"""Validate implementation plans against the documentation relationship graph."""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import json
import re
import subprocess
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Sequence

import yaml  # type: ignore[import-untyped]

from enforced_planning.file_context import collect_context
from enforced_planning.file_context import load_relationships
from enforced_planning.notebook_registry_validation import load_notebook_registry
from enforced_planning.notebook_registry_validation import print_human_readable
from enforced_planning.notebook_registry_validation import validate_notebook_registry
from enforced_planning.worktree_paths import detect_workspace_root


def _detect_repo_root(script_path: Path) -> Path:
    """Resolve repo root for canonical, installed, and package layouts."""
    if script_path.parent.name == "meta" and script_path.parent.parent.name == "scripts":
        return script_path.parents[2]
    if script_path.parent.name in {"scripts", "enforced_planning"}:
        return script_path.parents[1]
    return script_path.parents[1]


ROOT = _detect_repo_root(Path(__file__).resolve())
PLANS_DIR = ROOT / "docs" / "plans"
PATH_CLEAN_RE = re.compile(r"[,;:.()]$")


def normalize(path: str) -> str:
    """Normalize a repo-relative path for comparisons."""
    return str(path).replace("\\", "/").strip()


def get_current_plan_number(repo_root: Path = ROOT) -> int | None:
    """Infer the active plan number from the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    branch = result.stdout.strip()
    match = re.match(r".*plan-(\d+).*", branch)
    if not match:
        return None
    return int(match.group(1))


def find_plan_file(plan_number: int, plans_dir: Path) -> Path | None:
    """Find one plan file by numeric prefix inside the plans directory."""
    patterns = [
        f"{plan_number:02d}_*.md",
        f"{plan_number}_*.md",
    ]
    for pattern in patterns:
        matches = sorted(plans_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_text(path: Path) -> str:
    """Read one UTF-8 text file from disk."""
    return path.read_text(encoding="utf-8")


def extract_section(content: str, heading: str) -> str:
    """Extract one second-level markdown section body by heading name."""
    pattern = re.compile(
        rf"^##\s*{re.escape(heading)}\s*\n(.*?)(?=^##\s|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        return ""
    return match.group(1).strip()


def split_lines(section: str) -> list[str]:
    """Return non-empty stripped lines from one markdown section body."""
    lines: list[str] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(stripped)
    return lines


def looks_like_file_path(value: str) -> bool:
    """Heuristic filter for inline repo-relative file path references."""
    if not value:
        return False
    value = value.strip().strip("`\"'()[],")
    if value in {"None", "none", "N/A", "n/a", "-"}:
        return False
    if value.startswith("#") or value.startswith("http://") or value.startswith("https://"):
        return False
    return bool("/" in value or "\\" in value or re.search(r"\.[A-Za-z0-9]{1,8}$", value))


def extract_inline_paths(line: str) -> list[str]:
    """Extract path-like tokens from one line of markdown text."""
    paths: list[str] = []
    for match in re.finditer(r"\b([A-Za-z0-9_./-]+\.[A-Za-z0-9][A-Za-z0-9._-]*)\b", line):
        value = match.group(1)
        if looks_like_file_path(value):
            paths.append(normalize(value))

    for match in re.finditer(r"`([^`]+)`", line):
        value = match.group(1).strip()
        candidate = value.split(":", 1)[0] if ":" in value else value
        candidate = PATH_CLEAN_RE.sub("", candidate).strip()
        if looks_like_file_path(candidate):
            paths.append(normalize(candidate))

    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", line):
        value = match.group(1).strip()
        if looks_like_file_path(value):
            paths.append(normalize(value))

    return paths


def extract_paths(section: str) -> list[str]:
    """Extract and de-duplicate path references from one markdown section."""
    result: list[str] = []
    for line in split_lines(section):
        if line.startswith("|") and line.endswith("|") and re.match(r"^\|[-\s|:]+\|$", line):
            continue
        result.extend(extract_inline_paths(line))
    dedupe: list[str] = []
    seen: set[str] = set()
    for path in result:
        if path and path not in seen:
            seen.add(path)
            dedupe.append(path)
    return dedupe


def parse_files_affected(content: str) -> list[str]:
    """Parse the Files Affected section of a plan."""
    return extract_paths(extract_section(content, "Files Affected"))


def parse_references_reviewed(content: str) -> list[str]:
    """Parse the References Reviewed section of a plan."""
    return extract_paths(extract_section(content, "References Reviewed"))


def parse_uncertainty_register(content: str) -> list[str]:
    """Parse bullet items under the Uncertainty Register section."""
    section = extract_section(content, "Uncertainty Register")
    if not section:
        return []
    items: list[str] = []
    for line in split_lines(section):
        if line.startswith("- ") or line.startswith("* "):
            value = line[2:].strip()
            if value and value != "-":
                items.append(value)
    return items


def parse_contracts_used(content: str) -> list[str]:
    """Parse the Contracts Used section from a plan."""
    section = extract_section(content, "Contracts Used")
    if not section:
        return []
    contracts = []
    for line in section.split("\n"):
        line = line.strip()
        if line.startswith("- ") or line.startswith("* "):
            name = line[2:].split("—")[0].split("-")[0].strip().strip("`")
            if name and name != "-":
                contracts.append(name)
    return contracts


def parse_tools_used(content: str) -> list[str]:
    """Parse the Tools Used section from a plan."""
    section = extract_section(content, "Tools Used")
    if not section:
        return []
    tools = []
    for line in section.split("\n"):
        line = line.strip()
        if line.startswith("- ") or line.startswith("* "):
            name = line[2:].split("—")[0].split("-")[0].strip().strip("`")
            if name and name != "-":
                tools.append(name)
    return tools


def parse_data_flow(content: str) -> list[dict[str, str]]:
    """Parse the Data Flow section from a plan."""
    section = extract_section(content, "Data Flow")
    if not section:
        return []

    flows: list[dict[str, str]] = []
    for line in section.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if re.match(r"^\|[-\s|:]+\|$", line):
            continue
        if re.match(r"^\|\s*(Step|#)", line, re.IGNORECASE):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 6:
            flows.append({
                "producer": parts[2].strip("`"),
                "producer_schema": parts[3].strip("`"),
                "consumer": parts[4].strip("`"),
                "consumer_schema": parts[5].strip("`"),
            })
    return flows


def parse_plan_status(content: str) -> tuple[str, str]:
    """Extract the markdown title and bolded Status line."""
    status = "Unknown"
    if m := re.search(r"\*{1,2}Status:?\*?\s*:?\s*([^\n]+)", content, re.IGNORECASE):
        status = m.group(1).strip()
    title = "Plan"
    if first := re.search(r"^#\s*([^\n]+)", content, re.MULTILINE):
        title = first.group(1).strip()
    return title, status


def parse_mentioned_adrs(content: str) -> set[int]:
    """Return ADR numbers mentioned anywhere in the plan text."""
    result: set[int] = set()
    for match in re.finditer(r"\bADR[-_](\d{1,4})\b", content, re.IGNORECASE):
        try:
            result.add(int(match.group(1)))
        except ValueError:
            continue
    return result


def get_plan_file(
    plan_number: int | None,
    plans_dir: Path,
    plan_file: str | None,
    *,
    repo_root: Path = ROOT,
) -> Path:
    """Resolve the effective plan file from CLI inputs."""
    if plan_file:
        candidate = Path(plan_file)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        if not candidate.exists():
            print(f"Plan file not found: {candidate}")
            raise SystemExit(1)
        return candidate

    if plan_number is None:
        print("No plan specified and branch name does not contain plan-<N>.")
        print("Use --plan <N> or --plan-file <path>.")
        raise SystemExit(1)

    plan_path = find_plan_file(plan_number, plans_dir)
    if not plan_path:
        print(f"Plan file not found for plan #{plan_number}")
        raise SystemExit(1)
    return plan_path


def collect_plan_requirements(
    file_paths: list[str],
    relationships: dict[str, Any],
) -> tuple[set[str], set[str], set[int], list[tuple[str, int, str]]]:
    """Collect required docs and ADRs implied by affected files."""
    required_docs_strict: set[str] = set()
    required_docs_soft: set[str] = set()
    required_adrs: dict[int, str] = {}
    governance: list[tuple[str, int, str]] = []

    for file_path in file_paths:
        ctx = collect_context(file_path, relationships)
        if not ctx.governance and not ctx.current_arch_docs and not ctx.coupled_docs:
            continue

        for adr in ctx.governance:
            required_adrs[adr["adr"]] = adr.get("title", f"ADR-{adr['adr']:04d}")
            governance.append((adr["path"], adr["adr"], adr["title"]))

        required_docs_strict.update(ctx.current_arch_docs)
        required_docs_strict.update(ctx.target_arch_docs)
        required_docs_strict.update(ctx.gap_docs)
        required_docs_strict.update(ctx.plan_refs)

        for coupling in ctx.coupled_docs:
            target = normalize(coupling["path"])
            if coupling.get("soft"):
                required_docs_soft.add(target)
            else:
                required_docs_strict.add(target)

    return required_docs_strict, required_docs_soft, set(required_adrs), governance


REQUIRED_PLAN_SECTIONS: dict[str, tuple[list[str], str]] = {
    "Gap": (
        ["Gap", "Goal", "Problem"],
        "Must describe what exists now and what we want (requirements).",
    ),
    "Research": (
        ["Research", "References Reviewed", "References", "Prior Art"],
        "Must cite code/docs/prior art reviewed before planning (no guessing).",
    ),
    "Acceptance Criteria": (
        ["Acceptance Criteria", "Verification", "Success Criteria"],
        "Must define verifiable criteria for completion.",
    ),
}


@dataclass
class ValidationResult:
    """Structured output for one plan validation run."""

    plan_number: int | None
    plan_file: Path
    title: str
    status: str
    affected_files: list[str]
    references_reviewed: list[str]
    uncertainties: list[str]
    required_docs_strict: set[str]
    required_docs_soft: set[str]
    missing_strict: set[str]
    missing_soft: set[str]
    missing_adrs: list[tuple[int, str]]
    governance: list[tuple[str, int, str]]
    missing_sections: list[tuple[str, str]]
    data_flow: list[dict[str, str]]
    contracts_used: list[str]
    tools_used: list[str]

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable summary payload."""
        return {
            "plan_number": self.plan_number,
            "plan_file": str(self.plan_file),
            "title": self.title,
            "status": self.status,
            "affected_files": self.affected_files,
            "references_reviewed": self.references_reviewed,
            "uncertainties_count": len(self.uncertainties),
            "required_docs": {
                "strict": sorted(self.required_docs_strict),
                "soft": sorted(self.required_docs_soft),
                "missing_strict": sorted(self.missing_strict),
                "missing_soft": sorted(self.missing_soft),
            },
            "governance": [
                {"source": source, "adr": adr, "title": title}
                for source, adr, title in self.governance
            ],
            "missing_adrs": [
                {"adr": adr, "title": title}
                for adr, title in self.missing_adrs
            ],
            "data_flow": self.data_flow,
            "contracts_used": self.contracts_used,
            "tools_used": self.tools_used,
            "missing_sections": [
                {"section": name, "reason": reason}
                for name, reason in self.missing_sections
            ],
        }


def validate_plan(plan_file: Path, plan_number: int | None, relationships: dict[str, Any]) -> ValidationResult:
    """Validate one plan file against required docs, ADRs, and section rules."""
    content = read_text(plan_file)
    title, status = parse_plan_status(content)
    affected = parse_files_affected(content)
    if not affected:
        affected = extract_paths(extract_section(content, "Task Pack"))

    references = parse_references_reviewed(content)
    uncertainties = parse_uncertainty_register(content)
    covered = {normalize(p) for p in set(affected) | set(references)}

    required_strict, required_soft, adr_nums, governance = collect_plan_requirements(
        affected,
        relationships,
    )
    required_strict_norm = {normalize(p) for p in required_strict}
    required_soft_norm = {normalize(p) for p in required_soft}

    mentioned_adrs = parse_mentioned_adrs(content)
    adrs_meta: dict[int, dict[str, Any]] = {
        int(k): v for k, v in relationships.get("adrs", {}).items()
    }

    missing_adrs: list[tuple[int, str]] = []
    for adr_num in sorted(adr_nums):
        adr_ref = f"ADR-{adr_num:04d}"
        if adr_num in mentioned_adrs:
            continue
        adr_meta = adrs_meta.get(adr_num)
        adr_file = normalize(adr_meta.get("file", "")) if adr_meta else ""
        if adr_file and adr_file not in covered:
            adr_title = str(adr_meta.get("title", adr_ref)) if adr_meta else adr_ref
            missing_adrs.append((adr_num, adr_title))

    missing_strict = {path for path in required_strict_norm if path not in covered}
    missing_soft = {path for path in required_soft_norm if path not in covered}

    data_flow = parse_data_flow(content)
    contracts_used = parse_contracts_used(content)
    tools_used = parse_tools_used(content)

    missing_sections: list[tuple[str, str]] = []
    for section_name, (aliases, reason) in REQUIRED_PLAN_SECTIONS.items():
        found = False
        for alias in aliases:
            section_content = extract_section(content, alias)
            if section_content and len(section_content.strip()) >= 10:
                found = True
                break
        if not found:
            missing_sections.append((section_name, reason))

    return ValidationResult(
        plan_number=plan_number,
        plan_file=plan_file,
        title=title,
        status=status,
        affected_files=sorted(affected),
        references_reviewed=sorted(references),
        uncertainties=uncertainties,
        required_docs_strict=required_strict_norm,
        required_docs_soft=required_soft_norm,
        missing_strict=missing_strict,
        missing_soft=missing_soft,
        missing_adrs=missing_adrs,
        governance=governance,
        missing_sections=missing_sections,
        data_flow=data_flow,
        contracts_used=contracts_used,
        tools_used=tools_used,
    )


def print_summary(result: ValidationResult) -> None:
    """Print a concise human-readable validation summary."""
    print(f"Plan validation: {result.title}")
    if result.plan_number is not None:
        print(f"  Number: #{result.plan_number}")
    print(f"  File: {result.plan_file}")
    print(f"  Status: {result.status}")
    print(f"  Affected files: {len(result.affected_files)}")
    if not result.affected_files:
        print("  (no affected files discovered)")
    for path in result.affected_files:
        print(f"    - {path}")

    print("\nGOVERNANCE:")
    if result.governance:
        seen = set()
        for source, adr, title in result.governance:
            key = (source, adr, title)
            if key in seen:
                continue
            seen.add(key)
            print(f"  - ADR-{adr:04d}: {title} ({source})")
    else:
        print("  (none)")

    print("\nREQUIRED DOCUMENTATION (strict):")
    if result.required_docs_strict:
        for path in sorted(result.required_docs_strict):
            print(f"  - {path}")
    else:
        print("  (none)")

    if result.required_docs_soft:
        print("\nREQUIRED DOCUMENTATION (soft):")
        for path in sorted(result.required_docs_soft):
            print(f"  - {path}")

    if result.missing_strict or result.missing_adrs or result.missing_soft:
        print("\nGAPS FOUND:")
        if result.missing_strict:
            print("  Strict documentation gaps:")
            for path in sorted(result.missing_strict):
                print(f"    - {path}")
        if result.missing_adrs:
            print("  Missing governance ADR coverage in plan text:")
            for adr_num, title in result.missing_adrs:
                print(f"    - ADR-{adr_num:04d}: {title}")
        if result.missing_soft:
            print("  Soft documentation gaps:")
            for path in sorted(result.missing_soft):
                print(f"    - {path}")
    else:
        print("\nNo documentation gaps found.")

    if result.contracts_used:
        print(f"\nCONTRACTS USED ({len(result.contracts_used)}):")
        for contract in result.contracts_used:
            print(f"  - {contract}")

    if result.tools_used:
        print(f"\nTOOLS USED ({len(result.tools_used)}):")
        for tool in result.tools_used:
            print(f"  - {tool}")

    if result.data_flow:
        print(f"\nDATA FLOW DECLARATIONS ({len(result.data_flow)} boundary crossings):")
        for flow in result.data_flow:
            print(
                f"  {flow['producer']} ({flow['producer_schema']}) → "
                f"{flow['consumer']} ({flow['consumer_schema']})"
            )

    if result.missing_sections:
        print("\nMISSING PLAN SECTIONS (design sequence enforcement):")
        for section_name, reason in result.missing_sections:
            print(f"  - {section_name}: {reason}")

    if result.uncertainties:
        print(f"\nUncertainty register entries: {len(result.uncertainties)} (not a blocker)")


def load_ack_file(ack_path: Path) -> dict[str, str]:
    """Load acknowledged plan-validation gaps from a YAML file."""
    if not ack_path.exists():
        return {}
    try:
        data = yaml.safe_load(ack_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    acknowledgments: dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path", "")
        reason = str(entry.get("reason", "")).strip()
        if path and reason:
            acknowledgments[str(Path(path))] = reason
    return acknowledgments


def _apply_acknowledgments(
    result: ValidationResult,
    *,
    ack_file: str | None,
) -> set[str]:
    """Downgrade acknowledged strict gaps to non-blocking notes."""
    acknowledged: set[str] = set()
    if not ack_file:
        return acknowledged

    ack_path = Path(ack_file)
    if not ack_path.exists():
        return acknowledged

    acknowledgments = load_ack_file(ack_path)
    for missing_path in list(result.missing_strict):
        if missing_path in acknowledgments:
            acknowledged.add(missing_path)
            result.missing_strict.discard(missing_path)
            continue
        if any(
            fnmatch.fnmatch(ack_path_glob, missing_path)
            or fnmatch.fnmatch(missing_path, ack_path_glob)
            for ack_path_glob in acknowledgments
        ):
            acknowledged.add(missing_path)
            result.missing_strict.discard(missing_path)
    return acknowledged


def _run_soft_notebook_check(repo_root: Path) -> None:
    """Run the non-blocking notebook registry validation for one repo."""
    registry_path = repo_root / "notebooks" / "notebook_registry.yaml"
    if not registry_path.exists():
        return

    workspace_root = detect_workspace_root(repo_root)
    try:
        registry = load_notebook_registry(registry_path)
        notebook_result = validate_notebook_registry(
            registry,
            registry_path=registry_path,
            workspace_root=workspace_root,
        )
    except (ValueError, yaml.YAMLError) as exc:
        print("\nNOTEBOOK REGISTRY (soft check — non-blocking):")
        print(f"  notebook registry check error: {exc}")
        return

    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        print_human_readable(notebook_result)
    rendered = buffer.getvalue().strip()
    if rendered:
        print("\nNOTEBOOK REGISTRY (soft check — non-blocking):")
        for line in rendered.splitlines():
            print(f"  {line}")


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | None = None,
    plans_dir: Path | None = None,
) -> int:
    """Run plan validation from the command line."""
    base_repo_root = repo_root or ROOT
    base_plans_dir = plans_dir or (base_repo_root / "docs" / "plans")

    parser = argparse.ArgumentParser(
        description="Validate a plan against the documentation relationship graph"
    )
    parser.add_argument("--plan", type=int, help="Plan number (e.g. 28)")
    parser.add_argument("--plan-file", help="Plan file path (overrides --plan)")
    parser.add_argument(
        "--plans-dir",
        default=str(base_plans_dir),
        help="Directory containing plan files",
    )
    parser.add_argument(
        "--config",
        default="scripts/relationships.yaml",
        help="Path to relationship configuration",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Do not fail when gaps are found",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable result",
    )
    parser.add_argument(
        "--ack-file",
        default=None,
        help="Path to YAML file with acknowledged doc gaps (path + reason per entry)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    resolved_plans_dir = Path(args.plans_dir)
    if not resolved_plans_dir.is_absolute():
        resolved_plans_dir = base_repo_root / resolved_plans_dir

    plan_number = args.plan
    if plan_number is None and not args.plan_file:
        plan_number = get_current_plan_number(repo_root=base_repo_root)

    plan_path = get_plan_file(
        plan_number,
        resolved_plans_dir,
        args.plan_file,
        repo_root=base_repo_root,
    )
    if not args.plan and plan_number is None:
        match = re.match(r"(\d+)_", plan_path.name)
        if match:
            plan_number = int(match.group(1))

    relationships = load_relationships(
        repo_root=base_repo_root,
        config_path=args.config,
    )
    result = validate_plan(plan_path, plan_number, relationships)
    acknowledged = _apply_acknowledgments(result, ack_file=args.ack_file)

    if args.json:
        print(json.dumps(result.to_payload(), indent=2))
    else:
        print_summary(result)
        if acknowledged:
            print("\nACKNOWLEDGED GAPS (non-blocking):")
            ack_data = load_ack_file(Path(args.ack_file)) if args.ack_file else {}
            for path in sorted(acknowledged):
                reason = next(
                    (
                        ack_data[ack_path]
                        for ack_path in ack_data
                        if fnmatch.fnmatch(ack_path, path) or fnmatch.fnmatch(path, ack_path)
                    ),
                    "acknowledged",
                )
                print(f"  - {path} — {reason}")

    if args.warn_only:
        return 0

    if result.missing_strict or result.missing_adrs or result.missing_sections:
        return 1

    if not args.json:
        _run_soft_notebook_check(base_repo_root)

    return 0
