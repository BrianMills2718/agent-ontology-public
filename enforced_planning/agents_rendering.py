"""Importable AGENTS.md rendering helpers for governed repos."""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


SECTION_RE = re.compile(
    r"^##\s+(?P<heading>[^\n]+)\n(?P<body>.*?)(?=^##\s|\Z)",
    re.MULTILINE | re.DOTALL,
)

SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "Commands": ("Commands", "Quick Reference - Commands"),
    "Principles": ("Principles", "Design Principles"),
    "Workflow": ("Workflow",),
    "References": ("References",),
}


@dataclass(frozen=True)
class CanonicalInputs:
    """Resolved canonical inputs used to render ``AGENTS.md``."""

    repo_root: Path
    claude_path: Path
    relationships_path: Path
    output_path: Path
    template_path: Path


@dataclass(frozen=True)
class RendererRuntime:
    """Runtime configuration for one truthful render entrypoint layout."""

    script_path: Path
    repo_root: Path
    default_template: Path

    def repo_relative(self, path: Path, repo_root: Path) -> str:
        """Return a stable path string for generated output provenance."""

        resolved = path.resolve()
        repo_root_resolved = repo_root.resolve()
        canonical_repo_root = self.repo_root.resolve()
        if resolved.is_relative_to(repo_root_resolved):
            return str(resolved.relative_to(repo_root_resolved))
        if resolved.is_relative_to(canonical_repo_root):
            return str(resolved.relative_to(canonical_repo_root))
        return path.name

    def resolve_inputs(
        self,
        repo_root: Path,
        claude_file: str = "CLAUDE.md",
        relationships_file: str = "scripts/relationships.yaml",
        output_file: str = "AGENTS.md",
        template_path: Path | None = None,
    ) -> CanonicalInputs:
        """Resolve canonical input paths and fail loudly when missing."""

        if template_path is None:
            template_path = self.default_template

        claude_path = repo_root / claude_file
        relationships_path = repo_root / relationships_file
        output_path = repo_root / output_file

        if not claude_path.exists():
            raise FileNotFoundError(f"Missing canonical governance file: {claude_path}")
        if not relationships_path.exists():
            raise FileNotFoundError(
                f"Missing machine-readable governance file: {relationships_path}"
            )
        if not template_path.exists():
            raise FileNotFoundError(f"Missing AGENTS template: {template_path}")
        if output_path.is_symlink() and output_path.resolve() == claude_path.resolve():
            raise ValueError(
                "AGENTS output path is a symlink to CLAUDE.md. Remove the symlink "
                "before rendering so generated output cannot overwrite canonical governance."
            )

        return CanonicalInputs(
            repo_root=repo_root,
            claude_path=claude_path,
            relationships_path=relationships_path,
            output_path=output_path,
            template_path=template_path,
        )

    def render_agents_markdown(self, inputs: CanonicalInputs) -> str:
        """Render the generated ``AGENTS.md`` content for a repo."""

        claude_text = inputs.claude_path.read_text(encoding="utf-8")
        relationships_text = inputs.relationships_path.read_text(encoding="utf-8")
        template_text = inputs.template_path.read_text(encoding="utf-8")
        relationships_sha256 = hashlib.sha256(
            relationships_text.encode("utf-8")
        ).hexdigest()[:12]

        generator_relpath = self.repo_relative(self.script_path, inputs.repo_root)
        sync_checker_relpath = self.repo_relative(
            self.script_path.with_name("check_agents_sync.py"),
            inputs.repo_root,
        )
        claude_relpath = self.repo_relative(inputs.claude_path, inputs.repo_root)
        relationships_relpath = self.repo_relative(
            inputs.relationships_path, inputs.repo_root
        )

        machine_governance_note = (
            f"`{relationships_relpath}` is the source of truth for machine-readable "
            "governance in this repo: ADR coupling, required-reading edges, and "
            "doc-code linkage. This generated file does not inline that graph; it "
            "records the canonical path and sync marker, then points operators and "
            "validators back to the source graph. Prefer deterministic validators "
            "over prompt-only memory when those scripts are available."
        )

        rendered = template_text.format(
            title=extract_title(claude_text),
            generator_relpath=generator_relpath,
            sync_checker_relpath=sync_checker_relpath,
            claude_relpath=claude_relpath,
            relationships_relpath=relationships_relpath,
            relationships_sha256=relationships_sha256,
            overview=extract_overview(claude_text),
            commands=extract_section(claude_text, "Commands"),
            principles=extract_section(claude_text, "Principles"),
            workflow=extract_section(claude_text, "Workflow"),
            references=extract_section(claude_text, "References"),
            machine_governance_note=machine_governance_note,
        )
        return rendered.strip() + "\n"


def detect_repo_root(script_path: Path) -> Path:
    """Resolve the target repo root for both canonical and installed entrypoints."""

    if script_path.parent.name == "meta" and script_path.parent.parent.name == "scripts":
        return script_path.parents[2]
    if script_path.parent.name == "scripts":
        return script_path.parents[1]
    return script_path.parents[1]


def default_template_path(repo_root: Path) -> Path:
    """Return the truthful default template path for this repo layout."""

    candidates = (
        repo_root / "templates" / "agents.md.template",
        repo_root / "meta-process" / "templates" / "agents.md.template",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_renderer(script_path: Path) -> RendererRuntime:
    """Build one renderer runtime for a concrete entrypoint path."""

    repo_root = detect_repo_root(script_path)
    return RendererRuntime(
        script_path=script_path,
        repo_root=repo_root,
        default_template=default_template_path(repo_root),
    )


def extract_title(markdown: str) -> str:
    """Return the first H1 heading from a markdown document."""

    match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    if not match:
        raise ValueError("CLAUDE.md is missing a top-level title")
    return match.group(1).strip()


def extract_overview(markdown: str) -> str:
    """Return the intro block between the title and the first thematic break."""

    lines = markdown.splitlines()
    title_seen = False
    collected: list[str] = []
    for line in lines:
        if not title_seen:
            if line.startswith("# "):
                title_seen = True
            continue
        if line.strip() == "---" or line.startswith("## "):
            break
        collected.append(line)

    overview = "\n".join(collected).strip()
    if not overview:
        return (
            f"{extract_title(markdown)} uses `CLAUDE.md` as canonical repo "
            "governance and workflow policy."
        )
    return overview


def extract_section(markdown: str, heading: str) -> str:
    """Return the body for a named H2 section or supported heading alias."""

    accepted_headings = SECTION_ALIASES.get(heading, (heading,))
    for match in SECTION_RE.finditer(markdown):
        current_heading = match.group("heading").strip()
        if current_heading in accepted_headings:
            lines = match.group("body").splitlines()
            while lines and not lines[-1].strip():
                lines.pop()
            while lines and lines[-1].strip() == "---":
                lines.pop()
                while lines and not lines[-1].strip():
                    lines.pop()
            body = "\n".join(lines).strip()
            if not body:
                raise ValueError(f"CLAUDE.md section {current_heading!r} is empty")
            return body
    raise ValueError(
        "CLAUDE.md is missing required section: "
        f"{heading} (accepted headings: {', '.join(accepted_headings)})"
    )


def parse_args(default_template: Path) -> argparse.Namespace:
    """Parse CLI arguments for the renderer."""

    parser = argparse.ArgumentParser(
        description="Render a generated AGENTS.md from canonical repo governance",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repo root containing CLAUDE.md and scripts/relationships.yaml",
    )
    parser.add_argument(
        "--claude-file",
        default="CLAUDE.md",
        help="Repo-relative path to canonical CLAUDE.md",
    )
    parser.add_argument(
        "--relationships-file",
        default="scripts/relationships.yaml",
        help="Repo-relative path to canonical relationships.yaml",
    )
    parser.add_argument(
        "--output-file",
        default="AGENTS.md",
        help="Repo-relative output path for generated AGENTS.md",
    )
    parser.add_argument(
        "--template",
        default=str(default_template),
        help="Path to the AGENTS markdown template",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print rendered markdown instead of writing the output file",
    )
    return parser.parse_args()


__all__ = [
    "CanonicalInputs",
    "RendererRuntime",
    "build_renderer",
    "default_template_path",
    "detect_repo_root",
    "extract_overview",
    "extract_section",
    "extract_title",
    "parse_args",
]
