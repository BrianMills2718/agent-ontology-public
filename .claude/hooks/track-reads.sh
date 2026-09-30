#!/bin/bash
# Track file reads for required-reading enforcement.
# PostToolUse/Read hook — records each read in the session file and emits a
# structured hook log entry for observability.
#
# The session file is checked by gate-edit.sh to verify required docs were read
# before editing coupled files.

set -euo pipefail

resolve_repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || pwd
}

normalize_repo_path() {
    local raw_path="$1"
    local rel_path="$raw_path"
    if [[ "$raw_path" == "$REPO_ROOT/"* ]]; then
        rel_path="${raw_path#$REPO_ROOT/}"
    fi
    if [[ "$rel_path" == worktrees/* ]]; then
        rel_path="$(echo "$rel_path" | sed 's|^worktrees/[^/]*/||')"
    fi
    printf '%s' "$rel_path"
}

resolve_data_path() {
    local raw_path="$1"
    if [[ "$raw_path" == /* ]]; then
        printf '%s' "$raw_path"
    else
        printf '%s' "$REPO_ROOT/$raw_path"
    fi
}

log_read_event() {
    if [[ ! -f "$HOOK_LOG_SCRIPT" ]]; then
        return 0
    fi
    local -a command=(
        python "$HOOK_LOG_SCRIPT" read
        --file-path "$REL_PATH"
        --reads-file "$READS_FILE"
        --reason "read observed"
        --log-file "$LOG_FILE"
    )
    if [[ -n "${HOOK_EXPERIMENT_ID:-}" ]]; then
        command+=(--experiment-id "$HOOK_EXPERIMENT_ID")
    fi
    if [[ -n "${HOOK_VARIANT_ID:-}" ]]; then
        command+=(--variant-id "$HOOK_VARIANT_ID")
    fi
    if [[ -n "${HOOK_DOWNSTREAM_RUN_ID:-}" ]]; then
        command+=(--downstream-run-id "$HOOK_DOWNSTREAM_RUN_ID")
    fi
    "${command[@]}" \
        >/dev/null
}

INPUT="$(cat)"
FILE_PATH="$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null || echo "")"

if [[ -z "$FILE_PATH" ]]; then
    exit 0
fi

REPO_ROOT="$(resolve_repo_root)"
REL_PATH="$(normalize_repo_path "$FILE_PATH")"
READS_FILE="$(resolve_data_path "${CLAUDE_SESSION_READS_FILE:-/tmp/.claude_session_reads}")"
LOG_FILE="$(resolve_data_path "${CLAUDE_HOOK_LOG_FILE:-.claude/hook_log.jsonl}")"
HOOK_LOG_SCRIPT="$REPO_ROOT/scripts/meta/hook_log.py"
HOOK_EXPERIMENT_ID="${CLAUDE_HOOK_EXPERIMENT_ID:-}"
HOOK_VARIANT_ID="${CLAUDE_HOOK_VARIANT_ID:-}"
HOOK_DOWNSTREAM_RUN_ID="${CLAUDE_HOOK_DOWNSTREAM_RUN_ID:-}"

mkdir -p "$(dirname "$READS_FILE")"
echo "$REL_PATH" >> "$READS_FILE"
log_read_event

exit 0
