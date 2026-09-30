"""Load and print context relationships for files from scripts/relationships.yaml.

This supports planning and implementation guardrails by surfacing:
- governing ADRs
- coupled documentation
- current/target architecture docs
- gap-analysis docs
- scoped-file policy outcomes used by the read-gating hooks
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("scripts/relationships.yaml")
DEFAULT_READS_FILE = Path("/tmp/.claude_session_reads")


def _normalize(path: str) -> str:
    return str(path).replace("\\", "/")


def _to_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def _parse_adr(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if text.upper().startswith("ADR-"):
        text = text[4:]
    try:
        return int(text)
    except ValueError:
        return None


def _match(patterns: list[str], candidate: str) -> bool:
    normalized = _normalize(candidate)
    for pattern in patterns:
        if not pattern:
            continue
        normalized_pattern = _normalize(str(pattern).strip())
        if fnmatch.fnmatch(normalized, normalized_pattern):
            return True
        if normalized_pattern.endswith("/**"):
            base = normalized_pattern[:-3]
            if normalized == base or normalized.startswith(base + "/"):
                return True
        elif normalized_pattern.endswith("/"):
            base = normalized_pattern[:-1]
            if normalized == base or normalized.startswith(base + "/"):
                return True
        if fnmatch.fnmatch(normalized.split("/")[-1], normalized_pattern):
            return True
    return False


def _normalize_scope_mode(
    raw: Any,
    *,
    default: str,
) -> str:
    """Normalize a file-scope policy mode into canonical values."""
    if raw is None:
        return default

    value = str(raw).strip().lower().replace("_", "-")
    if value in {"hard-fail", "hardfail", "fail", "blocked", "block", "deny"}:
        return "hard-fail"
    if value in {"warn", "warning", "warn-only", "soft"}:
        return "warn"
    if value in {"ignore", "allow", "pass", "skip"}:
        return "ignore"
    return default


def _get_file_scope_config(relationships: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize file_scope settings from relationships config."""
    raw_scope = relationships.get("file_scope")
    if not isinstance(raw_scope, dict) or not raw_scope:
        return {
            "enabled": False,
            "managed_include": [],
            "managed_exclude": [],
            "managed_mode": "ignore",
            "unmanaged_mode": "ignore",
        }

    managed_section = raw_scope.get("managed")
    if isinstance(managed_section, dict):
        include = _to_list(managed_section.get("include") or managed_section.get("includes"))
        exclude = _to_list(managed_section.get("exclude") or managed_section.get("excludes"))
        managed_mode_raw = (
            managed_section.get("mode")
            or managed_section.get("policy")
            or managed_section.get("on_unregistered")
            or managed_section.get("unregistered")
        )
    else:
        include = _to_list(raw_scope.get("include") or raw_scope.get("includes"))
        exclude = _to_list(raw_scope.get("exclude") or raw_scope.get("excludes"))
        managed_mode_raw = (
            raw_scope.get("mode")
            or raw_scope.get("policy")
            or raw_scope.get("on_unregistered")
            or raw_scope.get("unregistered")
        )

    unmanaged_section = raw_scope.get("unmanaged")
    if isinstance(unmanaged_section, dict):
        unmanaged_mode_raw = (
            unmanaged_section.get("mode")
            or unmanaged_section.get("policy")
            or unmanaged_section.get("on_unregistered")
            or unmanaged_section.get("unregistered")
        )
    else:
        unmanaged_mode_raw = raw_scope.get(
            "unmanaged_mode",
            raw_scope.get("unmanaged"),
        )

    return {
        "enabled": True,
        "managed_include": include or [],
        "managed_exclude": exclude or [],
        "managed_mode": _normalize_scope_mode(
            managed_mode_raw,
            default="hard-fail",
        ),
        "unmanaged_mode": _normalize_scope_mode(
            unmanaged_mode_raw,
            default="ignore",
        ),
    }


def _is_managed_file(file_path: str, scope_config: dict[str, Any]) -> bool:
    """Whether a file path is within managed scope for gating policy."""
    if not scope_config.get("enabled"):
        return False

    include_patterns = scope_config.get("managed_include") or ["**"]
    exclude_patterns = scope_config.get("managed_exclude") or []

    if not _match(include_patterns, file_path):
        return False
    if exclude_patterns and _match(exclude_patterns, file_path):
        return False
    return True


@dataclass
class ReadCheckResult:
    """Result container for `check_required_reads` with optional scope policy metadata."""

    ok: bool
    required_reads: list[str]
    missing_reads: list[str]
    scope_violations: list[str]
    scope_warnings: list[str]

    def __iter__(self):
        """Yield the three legacy fields for unpack compatibility."""
        return iter((self.ok, self.required_reads, self.missing_reads))


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_adrs(raw_adrs: Any) -> dict[int, dict[str, Any]]:
    if isinstance(raw_adrs, dict):
        result: dict[int, dict[str, Any]] = {}
        for key, value in raw_adrs.items():
            adr_num = _parse_adr(key)
            if adr_num is None:
                continue
            if isinstance(value, dict):
                entry = dict(value)
            else:
                entry = {"title": str(value)}
            entry["title"] = str(entry.get("title", f"ADR-{adr_num:04d}"))
            entry["file"] = str(entry.get("file", ""))
            result[adr_num] = entry
        return result
    return {}


def _migrate_legacy_governance(raw: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        return result
    for entry in raw.get("governance", []) or []:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source") or entry.get("adr_file")
        applies_to = _to_list(entry.get("applies_to"))
        adr_num = _parse_adr(entry.get("adr"))
        if not applies_to:
            applies_to = _to_list(source)
        for source_path in applies_to:
            if adr_num is None:
                continue
            result.append({
                "source": source_path,
                "adrs": [adr_num],
                "context": f"ADR-{adr_num:04d}: {entry.get('title', '')}".strip(": "),
            })
    return result


def load_relationships(
    repo_root: Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = repo_root or REPO_ROOT
    if config_path is None:
        config_path = DEFAULT_CONFIG
    rel_path = Path(config_path)
    if not rel_path.is_absolute():
        rel_path = root / rel_path

    if rel_path.exists():
        relationships = load_yaml(rel_path)
    else:
        legacy = root / "scripts" / "doc_coupling.yaml"
        legacy_legacy = root / "scripts" / "governance.yaml"
        if legacy.exists():
            relationships = load_yaml(legacy)
        elif legacy_legacy.exists():
            relationships = load_yaml(legacy_legacy)
            relationships = {
                "couplings": relationships.get("couplings", []),
                "governance": _migrate_legacy_governance(relationships),
                "adrs": {},
            }
        else:
            return {"governance": [], "couplings": [], "adrs": {}, "architecture": []}

    relationships = dict(relationships or {})

    if "governance" not in relationships:
        if isinstance(relationships.get("files"), dict):
            flattened = []
            for source, info in relationships["files"].items():
                cfg = info if isinstance(info, dict) else {}
                flat = dict(cfg)
                flat["source"] = source
                flat.setdefault("adrs", flat.get("adrs", []))
                flat.setdefault("context", "")
                flattened.append(flat)
            relationships["governance"] = flattened
        else:
            relationships["governance"] = []

    relationships["governance"] = relationships["governance"] or []
    relationships["couplings"] = relationships["couplings"] or []
    relationships["architecture"] = relationships.get("architecture") or []
    relationships["adrs"] = _load_adrs(relationships.get("adrs"))
    relationships["file_scope"] = (
        relationships.get("file_scope")
        if isinstance(relationships.get("file_scope"), dict)
        else None
    )

    return relationships


@dataclass
class FileContext:
    path: str
    explicitly_registered: bool
    default_reads: list[str]
    governance: list[dict[str, Any]]
    coupled_docs: list[dict[str, Any]]
    current_arch_docs: list[str]
    target_arch_docs: list[str]
    gap_docs: list[str]
    plan_refs: list[str]

    @property
    def required_reads(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for doc in (
            self.default_reads
            + self.current_arch_docs
            + self.target_arch_docs
            + self.gap_docs
            + self.plan_refs
        ):
            normalized = _normalize(doc)
            if normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
        for coupled_doc in self.coupled_docs:
            normalized = _normalize(coupled_doc["path"])
            if normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.path,
            "governance": self.governance,
            "coupled_docs": [d["path"] for d in self.coupled_docs],
            "coupling_context": {
                d["path"]: d.get("description", "") for d in self.coupled_docs
            },
            "architecture": {
                "current": self.current_arch_docs,
                "target": self.target_arch_docs,
                "gaps": self.gap_docs,
                "plan_refs": self.plan_refs,
            },
            "required_reads": self.required_reads,
        }


def collect_context(
    file_path: str,
    relationships: dict[str, Any],
) -> FileContext:
    governance_cfg = relationships.get("governance", []) or []
    coupling_cfg = relationships.get("couplings", []) or []
    architecture_cfg = relationships.get("architecture", []) or []
    required_reading_cfg = relationships.get("required_reading", {}) or {}
    adrs = relationships.get("adrs", {}) or {}

    governance: list[dict[str, Any]] = []
    explicitly_registered = False
    for rule in governance_cfg:
        if not isinstance(rule, dict):
            continue
        patterns = _to_list(rule.get("source") or rule.get("sources"))
        if not _match(patterns, file_path):
            continue
        explicitly_registered = True
        for adr_num in _to_list(rule.get("adrs", [])):
            num = _parse_adr(adr_num)
            if num is None:
                continue
            context = {"path": _normalize(file_path), "adr": num}
            context["title"] = adrs.get(num, {}).get("title", f"ADR-{num:04d}")
            context["file"] = adrs.get(num, {}).get("file", "")
            context["notes"] = str(rule.get("context", "")).strip()
            governance.append(context)

    seen_docs: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for rule in coupling_cfg:
        if not isinstance(rule, dict):
            continue
        source_patterns = _to_list(rule.get("sources") or rule.get("source"))
        if not _match(source_patterns, file_path):
            continue
        explicitly_registered = True
        for doc in _to_list(rule.get("docs")):
            norm = _normalize(doc)
            if norm in seen_paths:
                continue
            seen_paths.add(norm)
            seen_docs.append({
                "path": norm,
                "description": str(rule.get("description", "")),
                "soft": bool(rule.get("soft")),
            })

    arch_current: list[str] = []
    arch_target: list[str] = []
    arch_gaps: list[str] = []
    arch_plan_refs: list[str] = []

    for rule in architecture_cfg:
        if not isinstance(rule, dict):
            continue
        source_patterns = _to_list(rule.get("source_patterns") or rule.get("sources") or rule.get("source"))
        if not _match(source_patterns, file_path):
            continue
        explicitly_registered = True
        arch_current.extend(_to_list(rule.get("current_docs") or rule.get("current")))
        arch_target.extend(_to_list(rule.get("target_docs") or rule.get("target")))
        arch_gaps.extend(_to_list(rule.get("gap_docs") or rule.get("gaps")))
        arch_plan_refs.extend(_to_list(rule.get("plan_refs")))

    def dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for value in values:
            norm = _normalize(value)
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
        return out

    return FileContext(
        path=_normalize(file_path),
        explicitly_registered=explicitly_registered,
        default_reads=dedupe(_to_list(required_reading_cfg.get("defaults"))),
        governance=governance,
        coupled_docs=seen_docs,
        current_arch_docs=dedupe(arch_current),
        target_arch_docs=dedupe(arch_target),
        gap_docs=dedupe(arch_gaps),
        plan_refs=dedupe(arch_plan_refs),
    )


def _load_reads(paths_file: Path) -> set[str]:
    if not paths_file.exists():
        return set()
    reads = set()
    for line in paths_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            reads.add(_normalize(line))
    return reads


def _any_seen_matches_glob(pattern: str, seen: set[str]) -> bool:
    """Check if any seen read matches a glob pattern in required reads."""
    if "*" not in pattern and "?" not in pattern:
        return False
    if "**" in pattern:
        parts = pattern.split("**")
        prefix = parts[0].rstrip("/")
        suffix_pattern = parts[-1].lstrip("/")
        for s in seen:
            if s.startswith(prefix) and fnmatch.fnmatch(s.split("/")[-1], suffix_pattern):
                return True
            remainder = s[len(prefix):].lstrip("/")
            if s.startswith(prefix) and fnmatch.fnmatch(remainder, suffix_pattern):
                return True
        return False
    return any(fnmatch.fnmatch(s, pattern) for s in seen)


def check_required_reads(
    file_path: str,
    relationships: dict[str, Any],
    reads_file: Path,
) -> ReadCheckResult:
    """Check whether file-specific required reads were completed for a given file."""
    context = collect_context(file_path, relationships)
    required = [path for path in context.required_reads if path != _normalize(file_path)]
    seen = _load_reads(reads_file)
    missing = [
        path for path in required
        if path not in seen and not _any_seen_matches_glob(path, seen)
    ]

    scope_config = _get_file_scope_config(relationships)
    scope_violations: list[str] = []
    scope_warnings: list[str] = []
    in_managed_scope = _is_managed_file(context.path, scope_config)
    if not context.explicitly_registered:
        if in_managed_scope:
            if scope_config["managed_mode"] == "hard-fail":
                scope_violations.append(context.path)
            elif scope_config["managed_mode"] == "warn":
                scope_warnings.append(context.path)
        elif scope_config["unmanaged_mode"] == "warn":
            scope_warnings.append(context.path)

    return ReadCheckResult(
        ok=(not missing and not scope_violations),
        required_reads=required,
        missing_reads=missing,
        scope_violations=scope_violations,
        scope_warnings=scope_warnings,
    )


def _render_summary(context: FileContext) -> str:
    lines: list[str] = [f"Context for {context.path}", ""]

    if context.governance:
        lines.append("GOVERNANCE:")
        for item in context.governance:
            adr = item["adr"]
            title = item["title"]
            file_ref = item.get("file", "")
            notes = item.get("notes", "")
            lines.append(f"  - ADR-{adr:04d}: {title}")
            if file_ref:
                lines.append(f"    source: {file_ref}")
            if notes:
                lines.append(f"    notes: {notes}")
    else:
        lines.append("GOVERNANCE: none")

    if context.coupled_docs:
        lines.append("DOCUMENT COUPLINGS:")
        for item in context.coupled_docs:
            lines.append(f"  - {item['path']}{' (soft)' if item.get('soft') else ''}")
            if item.get("description"):
                lines.append(f"    reason: {item['description']}")
    else:
        lines.append("DOCUMENT COUPLINGS: none")

    lines.append("ARCHITECTURE:")
    if context.current_arch_docs:
        lines.append("  current:")
        lines.extend(f"    - {d}" for d in context.current_arch_docs)
    if context.target_arch_docs:
        lines.append("  target:")
        lines.extend(f"    - {d}" for d in context.target_arch_docs)
    if context.gap_docs:
        lines.append("  gaps:")
        lines.extend(f"    - {d}" for d in context.gap_docs)
    if context.plan_refs:
        lines.append("  plan_refs:")
        lines.extend(f"    - {d}" for d in context.plan_refs)

    lines.append("REQUIRED READS:")
    if context.required_reads:
        lines.extend(f"  - {d}" for d in context.required_reads)
    else:
        lines.append("  (none)")

    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint for resolving file context and read requirements."""
    parser = argparse.ArgumentParser(
        description="Load file context from relationships.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="File(s) to resolve architecture/ADR/doc relationships for",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to relationships.yaml (legacy fallbacks supported)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON",
    )
    parser.add_argument(
        "--check-reads",
        action="store_true",
        help="Return non-zero if required reads have not been performed",
    )
    parser.add_argument(
        "--reads-file",
        default=str(DEFAULT_READS_FILE),
        help="File that records session read paths",
    )

    args = parser.parse_args()

    relationships = load_relationships(config_path=args.config)
    contexts: list[FileContext] = [collect_context(f, relationships) for f in args.files]

    if args.check_reads:
        overall_ok = True
        for context in contexts:
            check_result = check_required_reads(
                context.path,
                relationships,
                Path(args.reads_file),
            )
            required = check_result.required_reads
            missing = check_result.missing_reads
            if missing or check_result.scope_violations:
                overall_ok = False
                print(f"REQUIRED READS MISSING for {context.path}")
                print("  required:")
                for item in required:
                    print(f"    - {item}")
                print("  missing:")
                for item in missing:
                    print(f"    - {item}")
                if check_result.scope_violations:
                    print("  scope violations:")
                    for item in check_result.scope_violations:
                        print(f"    - {item}")
                if check_result.scope_warnings:
                    print("  scope warnings:")
                    for item in check_result.scope_warnings:
                        print(f"    - {item}")
            elif check_result.scope_warnings:
                print(f"SCOPE WARNINGS for {context.path}")
                print("  scope warnings:")
                for item in check_result.scope_warnings:
                    print(f"    - {item}")
        return 0 if overall_ok else 1

    if args.json:
        read_checks = [
            check_required_reads(
                context.path,
                relationships,
                Path(args.reads_file),
            )
            for context in contexts
        ]
        payload = {
            "files": [context.to_dict() for context in contexts],
            "required_reads": sorted(
                {
                    doc
                    for context in contexts
                    for doc in context.required_reads
                    for path in [doc]
                    if path not in args.files
                }
            ),
            "scope_violations": sorted(
                {
                    item
                    for result in read_checks
                    for item in result.scope_violations
                }
            ),
            "scope_warnings": sorted(
                {
                    item
                    for result in read_checks
                    for item in result.scope_warnings
                }
            ),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(_render_summary(contexts[0]))
    if len(contexts) > 1:
        for context in contexts[1:]:
            print("\n" + "-" * 79)
            print(_render_summary(context))
    return 0


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_READS_FILE",
    "FileContext",
    "ReadCheckResult",
    "check_required_reads",
    "collect_context",
    "load_relationships",
    "load_yaml",
    "main",
]

