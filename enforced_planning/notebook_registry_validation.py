"""Validate notebook registries and registered journey notebooks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import yaml  # type: ignore[import-untyped]

from enforced_planning.worktree_paths import detect_workspace_root


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = detect_workspace_root(REPO_ROOT)
DEFAULT_REGISTRY = REPO_ROOT / "notebooks" / "notebook_registry.yaml"

ALLOWED_JOURNEY_MODES = {"planning", "proof", "mixed"}
ALLOWED_PHASE_STATUS = {"planned", "partial", "proven"}
ALLOWED_EXECUTION_MODES = {"planned", "stub", "fixture", "dry_run", "live"}
HEADER_LABELS = (
    "Journey Name:",
    "Journey Purpose:",
    "Notebook Mode:",
    "Related Docs:",
    "Related Code:",
    "Related Tests:",
    "Related Evidence:",
)
PHASE_LABELS = (
    "Purpose:",
    "Input -> Output:",
    "Acceptance Criteria:",
    "Status:",
    "Execution Mode:",
)


@dataclass
class NotebookRegistryValidationResult:
    """Validation result for notebook registry checks."""

    registry_path: str
    journeys_checked: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True when no hard validation errors were found."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the result."""
        return asdict(self) | {"ok": self.ok}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML from disk and normalize empty documents to an empty mapping."""
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Registry root must be a mapping: {path}")
    return data


def load_notebook_registry(path: Path) -> dict[str, Any]:
    """Load the notebook registry YAML document from disk."""
    return load_yaml(path)


def resolve_workspace_path(raw_path: str, *, workspace_root: Path | None = None) -> Path:
    """Resolve a registry path relative to the shared workspace root."""
    base_root = workspace_root or WORKSPACE_ROOT
    return (base_root / raw_path).resolve()


def load_notebook(path: Path) -> dict[str, Any]:
    """Load and validate notebook JSON structure from disk."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Notebook root must be a JSON object: {path}")
    cells = data.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"Notebook must contain a list-valued 'cells' field: {path}")
    return data


def _to_list(value: Any) -> list[str]:
    """Normalize a scalar or list value into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _markdown_texts(notebook: dict[str, Any]) -> list[str]:
    """Return normalized markdown cell text for a notebook."""
    markdown_cells: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            markdown_cells.append("".join(str(part) for part in source))
        else:
            markdown_cells.append(str(source))
    return markdown_cells


def _validate_registry_top_level(
    registry: dict[str, Any],
    result: NotebookRegistryValidationResult,
) -> list[dict[str, Any]]:
    """Validate required registry top-level fields and return journey entries."""
    version = registry.get("version")
    if version != 1:
        result.errors.append(f"Registry version must be 1, found {version!r}.")

    journeys = registry.get("journeys")
    if not isinstance(journeys, list) or not journeys:
        result.errors.append("Registry must contain a non-empty 'journeys' list.")
        return []
    return [entry for entry in journeys if isinstance(entry, dict)]


def _validate_file_list(
    values: list[str],
    *,
    label: str,
    journey_id: str,
    result: NotebookRegistryValidationResult,
    workspace_root: Path,
) -> None:
    """Validate that each workspace-relative file path in a list exists."""
    for value in values:
        resolved = resolve_workspace_path(value, workspace_root=workspace_root)
        if not resolved.exists():
            result.errors.append(
                f"{journey_id}: {label} path does not exist: {value}"
            )


def _validate_notebook_header(
    notebook: dict[str, Any],
    *,
    journey: dict[str, Any],
    result: NotebookRegistryValidationResult,
) -> None:
    """Validate the visible header contract near the top of the notebook."""
    markdown_cells = _markdown_texts(notebook)
    header_text = "\n".join(markdown_cells[:3])
    journey_id = str(journey.get("journey_id", "unknown"))

    for label in HEADER_LABELS:
        if label not in header_text:
            result.errors.append(f"{journey_id}: notebook header missing label {label!r}.")

    if str(journey.get("title", "")) not in header_text:
        result.errors.append(f"{journey_id}: notebook header does not include the journey title.")
    if str(journey.get("notebook_mode", "")) not in header_text:
        result.errors.append(f"{journey_id}: notebook header does not include the notebook mode.")


def _validate_notebook_metadata(
    notebook: dict[str, Any],
    *,
    journey: dict[str, Any],
    phase_ids: list[str],
    result: NotebookRegistryValidationResult,
) -> None:
    """Validate notebook-level machine-readable metadata against the registry."""
    metadata = notebook.get("metadata", {})
    if not isinstance(metadata, dict):
        result.errors.append(
            f"{journey['journey_id']}: notebook metadata must be a mapping."
        )
        return

    journey_meta = metadata.get("journey_meta")
    if not isinstance(journey_meta, dict):
        result.errors.append(
            f"{journey['journey_id']}: notebook metadata missing 'journey_meta'."
        )
        return

    if journey_meta.get("journey_id") != journey.get("journey_id"):
        result.errors.append(
            f"{journey['journey_id']}: notebook metadata journey_id does not match registry."
        )
    if journey_meta.get("notebook_mode") != journey.get("notebook_mode"):
        result.errors.append(
            f"{journey['journey_id']}: notebook metadata notebook_mode does not match registry."
        )
    if journey_meta.get("phase_ids_in_order") != phase_ids:
        result.errors.append(
            f"{journey['journey_id']}: notebook metadata phase_ids_in_order does not match registry."
        )


def _validate_phase_sections(
    notebook: dict[str, Any],
    *,
    journey: dict[str, Any],
    phases: list[dict[str, Any]],
    result: NotebookRegistryValidationResult,
) -> None:
    """Validate that each registered phase has an explicit visible contract cell."""
    markdown_cells = _markdown_texts(notebook)
    journey_id = str(journey.get("journey_id", "unknown"))

    for phase in phases:
        title = str(phase.get("title", ""))
        status = str(phase.get("status", ""))
        execution_mode = str(phase.get("execution_mode", ""))
        matching_cell = next(
            (
                cell
                for cell in markdown_cells
                if "## Phase" in cell and title in cell
            ),
            None,
        )
        if matching_cell is None:
            result.errors.append(
                f"{journey_id}: notebook missing explicit section for phase {title!r}."
            )
            continue

        for label in PHASE_LABELS:
            if label not in matching_cell:
                result.errors.append(
                    f"{journey_id}: phase section {title!r} missing label {label!r}."
                )

        expected_status_line = f"Status: {status}"
        expected_mode_line = f"Execution Mode: {execution_mode}"
        if expected_status_line not in matching_cell:
            result.errors.append(
                f"{journey_id}: phase section {title!r} does not match registry status {status!r}."
            )
        if expected_mode_line not in matching_cell:
            result.errors.append(
                f"{journey_id}: phase section {title!r} does not match registry execution_mode {execution_mode!r}."
            )


def _validate_proof_mode(
    *,
    journey: dict[str, Any],
    phases: list[dict[str, Any]],
    result: NotebookRegistryValidationResult,
) -> None:
    """Enforce proof-mode semantics for proof-critical phases."""
    if journey.get("notebook_mode") != "proof":
        return

    for phase in phases:
        if not phase.get("proof_critical", False):
            continue
        if phase.get("status") != "proven" or phase.get("execution_mode") != "live":
            result.errors.append(
                f"{journey['journey_id']}: proof notebook has non-live or non-proven proof-critical "
                f"phase {phase.get('phase_id')!r}."
            )


def _validate_phase_entries(
    phases: list[dict[str, Any]],
    *,
    journey_id: str,
    result: NotebookRegistryValidationResult,
    workspace_root: Path,
) -> list[str]:
    """Validate phase entry shape and return phase IDs in declared order."""
    if not phases:
        result.errors.append(f"{journey_id}: journey must declare at least one phase.")
        return []

    seen_phase_ids: set[str] = set()
    phase_ids: list[str] = []
    for phase in phases:
        phase_id = str(phase.get("phase_id", "")).strip()
        if not phase_id:
            result.errors.append(f"{journey_id}: phase missing phase_id.")
            continue
        if phase_id in seen_phase_ids:
            result.errors.append(f"{journey_id}: duplicate phase_id {phase_id!r}.")
            continue
        seen_phase_ids.add(phase_id)
        phase_ids.append(phase_id)

        if phase.get("status") not in ALLOWED_PHASE_STATUS:
            result.errors.append(
                f"{journey_id}: phase {phase_id!r} has invalid status {phase.get('status')!r}."
            )
        if phase.get("execution_mode") not in ALLOWED_EXECUTION_MODES:
            result.errors.append(
                f"{journey_id}: phase {phase_id!r} has invalid execution_mode "
                f"{phase.get('execution_mode')!r}."
            )
        if not _to_list(phase.get("acceptance")):
            result.errors.append(f"{journey_id}: phase {phase_id!r} missing acceptance criteria.")
        if not str(phase.get("purpose", "")).strip():
            result.errors.append(f"{journey_id}: phase {phase_id!r} missing purpose.")
        if not str(phase.get("input_artifact", "")).strip():
            result.errors.append(f"{journey_id}: phase {phase_id!r} missing input_artifact.")
        if not str(phase.get("output_artifact", "")).strip():
            result.errors.append(f"{journey_id}: phase {phase_id!r} missing output_artifact.")

        for label in ("docs", "code", "tests", "evidence"):
            _validate_file_list(
                _to_list(phase.get(label)),
                label=f"phase {phase_id!r} {label}",
                journey_id=journey_id,
                result=result,
                workspace_root=workspace_root,
            )

    return phase_ids


def validate_notebook_registry(
    registry: dict[str, Any],
    *,
    registry_path: Path,
    journey_id: str | None = None,
    workspace_root: Path | None = None,
) -> NotebookRegistryValidationResult:
    """Validate the notebook registry and all selected journey entries."""
    base_workspace_root = workspace_root or WORKSPACE_ROOT
    result = NotebookRegistryValidationResult(registry_path=str(registry_path))
    journeys = _validate_registry_top_level(registry, result)
    if result.errors:
        return result

    seen_journey_ids: set[str] = set()
    seen_notebooks: set[str] = set()

    for journey in journeys:
        current_journey_id = str(journey.get("journey_id", "")).strip()
        if journey_id is not None and current_journey_id != journey_id:
            continue
        result.journeys_checked.append(current_journey_id)

        if not current_journey_id:
            result.errors.append("Journey entry missing journey_id.")
            continue
        if current_journey_id in seen_journey_ids:
            result.errors.append(f"Duplicate journey_id {current_journey_id!r}.")
            continue
        seen_journey_ids.add(current_journey_id)

        notebook_mode = journey.get("notebook_mode")
        if notebook_mode not in ALLOWED_JOURNEY_MODES:
            result.errors.append(
                f"{current_journey_id}: invalid notebook_mode {notebook_mode!r}."
            )

        notebook_rel = str(journey.get("notebook", "")).strip()
        if not notebook_rel:
            result.errors.append(f"{current_journey_id}: journey missing notebook path.")
            continue
        if notebook_rel in seen_notebooks:
            result.errors.append(
                f"{current_journey_id}: notebook path reused by another journey: {notebook_rel}"
            )
            continue
        seen_notebooks.add(notebook_rel)

        notebook_path = resolve_workspace_path(
            notebook_rel,
            workspace_root=base_workspace_root,
        )
        if not notebook_path.exists():
            result.errors.append(f"{current_journey_id}: notebook path does not exist: {notebook_rel}")
            continue

        for label in (
            "related_docs",
            "related_code",
            "related_tests",
            "related_evidence",
            "deep_dive_notebooks",
        ):
            _validate_file_list(
                _to_list(journey.get(label)),
                label=label,
                journey_id=current_journey_id,
                result=result,
                workspace_root=base_workspace_root,
            )

        phases = [entry for entry in journey.get("phases", []) if isinstance(entry, dict)]
        phase_ids = _validate_phase_entries(
            phases,
            journey_id=current_journey_id,
            result=result,
            workspace_root=base_workspace_root,
        )

        try:
            notebook = load_notebook(notebook_path)
        except (json.JSONDecodeError, ValueError) as exc:
            result.errors.append(f"{current_journey_id}: failed to load notebook {notebook_rel}: {exc}")
            continue

        _validate_notebook_metadata(
            notebook,
            journey=journey,
            phase_ids=phase_ids,
            result=result,
        )
        _validate_notebook_header(notebook, journey=journey, result=result)
        _validate_phase_sections(notebook, journey=journey, phases=phases, result=result)
        _validate_proof_mode(journey=journey, phases=phases, result=result)

        for deep_dive in _to_list(journey.get("deep_dive_notebooks")):
            deep_dive_path = resolve_workspace_path(
                deep_dive,
                workspace_root=base_workspace_root,
            )
            try:
                load_notebook(deep_dive_path)
            except (json.JSONDecodeError, ValueError) as exc:
                result.errors.append(
                    f"{current_journey_id}: failed to load deep-dive notebook {deep_dive}: {exc}"
                )

    if journey_id is not None and not result.journeys_checked:
        result.errors.append(f"Requested journey_id {journey_id!r} not found in registry.")

    return result


def print_human_readable(result: NotebookRegistryValidationResult) -> None:
    """Print a concise human-readable validation summary."""
    print(f"Notebook registry: {result.registry_path}")
    if result.journeys_checked:
        print("Journeys checked:")
        for journey in result.journeys_checked:
            print(f"  - {journey}")

    if result.errors:
        print("\nERRORS:")
        for error in result.errors:
            print(f"  - {error}")
    if result.warnings:
        print("\nWARNINGS:")
        for warning in result.warnings:
            print(f"  - {warning}")
    if result.ok:
        print("\nNotebook registry validation passed.")


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | None = None,
    workspace_root: Path | None = None,
) -> int:
    """Run notebook registry validation from the command line."""
    base_repo_root = repo_root or REPO_ROOT
    base_workspace_root = workspace_root or detect_workspace_root(base_repo_root)
    default_registry = base_repo_root / "notebooks" / "notebook_registry.yaml"

    parser = argparse.ArgumentParser(description="Validate notebook registry and journey notebooks")
    parser.add_argument(
        "--config",
        default=str(default_registry),
        help="Path to notebook registry YAML (default: <repo>/notebooks/notebook_registry.yaml)",
    )
    parser.add_argument(
        "--journey-id",
        default=None,
        help="Validate only one journey_id from the registry.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable output.",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Return exit code 0 even when validation errors are present.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    registry_path = Path(args.config)
    if not registry_path.is_absolute():
        registry_path = (base_repo_root / registry_path).resolve()
    if not registry_path.exists():
        print(f"Notebook registry not found: {registry_path}")
        return 1

    try:
        registry = load_notebook_registry(registry_path)
        result = validate_notebook_registry(
            registry,
            registry_path=registry_path,
            journey_id=args.journey_id,
            workspace_root=base_workspace_root,
        )
    except (ValueError, yaml.YAMLError) as exc:
        print(f"Failed to load notebook registry: {exc}")
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_human_readable(result)

    if result.ok or args.warn_only:
        return 0
    return 1
