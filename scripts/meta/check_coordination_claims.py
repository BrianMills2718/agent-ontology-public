#!/usr/bin/env python3
"""Compatibility facade for package-backed coordination claims."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _find_repo_root() -> Path:
    """Resolve the nearest ancestor that contains the installed support package."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "enforced_planning").is_dir():
            return parent
    raise RuntimeError("Unable to locate repo root containing enforced_planning/")


repo_root = _find_repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from enforced_planning import coordination_claims as _impl


CLAIMS_DIR = _impl.CLAIMS_DIR
DEFAULT_TTL_HOURS = _impl.DEFAULT_TTL_HOURS
LIVE_STATUSES = _impl.LIVE_STATUSES
CLAIM_TYPES = _impl.CLAIM_TYPES
STRICT_LIVE_METADATA_CLAIM_TYPES = _impl.STRICT_LIVE_METADATA_CLAIM_TYPES

ClaimRecord = _impl.ClaimRecord
ClaimInteraction = _impl.ClaimInteraction
ClaimCheckResult = _impl.ClaimCheckResult

_claim_filename = _impl._claim_filename
_normalize_repo_path = _impl._normalize_repo_path
_paths_overlap = _impl._paths_overlap


def _sync_runtime_config() -> None:
    """Keep the package module aligned with script-level monkeypatches."""
    _impl.CLAIMS_DIR = CLAIMS_DIR


def normalize_claim(data: dict[str, Any], *, source_file: str | None = None) -> ClaimRecord | None:
    """Delegate normalized claim loading to the package module."""
    return _impl.normalize_claim(data, source_file=source_file)


def check_claims(project: str | None = None) -> list[ClaimRecord]:
    """Delegate live-claim loading while honoring script-level CLAIMS_DIR overrides."""
    _sync_runtime_config()
    return _impl.check_claims(project)


def evaluate_claim(candidate: ClaimRecord, *, active_claims: list[ClaimRecord] | None = None) -> ClaimCheckResult:
    """Delegate candidate evaluation while honoring script-level CLAIMS_DIR overrides."""
    _sync_runtime_config()
    return _impl.evaluate_claim(candidate, active_claims=active_claims)


def build_candidate_claim(**kwargs: Any) -> ClaimRecord:
    """Delegate candidate claim construction to the package module."""
    return _impl.build_candidate_claim(**kwargs)


def claim_health_issues(claim: ClaimRecord) -> list[str]:
    """Expose claim health diagnostics through the legacy script surface."""
    return _impl.claim_health_issues(claim)


def claim_health_status(claim: ClaimRecord) -> str:
    """Expose claim health classification through the legacy script surface."""
    return _impl.claim_health_status(claim)


def claim_lifecycle_issues(claim: ClaimRecord) -> list[str]:
    """Expose stale-lifecycle diagnostics through the legacy script surface."""
    return _impl.claim_lifecycle_issues(claim)


def claim_runtime_status(claim: ClaimRecord) -> str:
    """Expose combined stale/weak/healthy runtime classification."""
    return _impl.claim_runtime_status(claim)


def claim_liveness_issues(claim: ClaimRecord, *, now: Any | None = None) -> list[str]:
    """Expose heartbeat-backed liveness diagnostics."""
    return _impl.claim_liveness_issues(claim, now=now)


def hydrate_missing_session_ids(*args: Any, **kwargs: Any) -> tuple[int, list[str], str]:
    """Delegate session-id hydration while honoring script-level CLAIMS_DIR overrides."""
    _sync_runtime_config()
    return _impl.hydrate_missing_session_ids(*args, **kwargs)


def create_claim(*args: Any, **kwargs: Any) -> tuple[bool, str]:
    """Delegate claim creation while honoring script-level CLAIMS_DIR overrides."""
    _sync_runtime_config()
    return _impl.create_claim(*args, **kwargs)


def release_claim(*args: Any, **kwargs: Any) -> tuple[bool, str]:
    """Delegate claim release while honoring script-level CLAIMS_DIR overrides."""
    _sync_runtime_config()
    return _impl.release_claim(*args, **kwargs)


def prune_expired() -> int:
    """Delegate claim pruning while honoring script-level CLAIMS_DIR overrides."""
    _sync_runtime_config()
    return _impl.prune_expired()


def prune_stale() -> tuple[int, list[str]]:
    """Delegate stale-claim pruning while honoring script-level CLAIMS_DIR overrides."""
    _sync_runtime_config()
    return _impl.prune_stale()


def heartbeat_claims(*args: Any, **kwargs: Any) -> tuple[int, list[str], str, str]:
    """Delegate heartbeat refresh while honoring script-level CLAIMS_DIR overrides."""
    _sync_runtime_config()
    return _impl.heartbeat_claims(*args, **kwargs)


def parse_args(argv: list[str] | None = None):
    """Expose CLI argument parsing through the legacy script surface."""
    return _impl.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the package-backed coordination-claims CLI."""
    _sync_runtime_config()
    return _impl.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
