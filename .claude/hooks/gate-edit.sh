#!/bin/bash
# Gate edits on required reading.
# PreToolUse/Edit hook — blocks edits when relationships-derived required docs
# have not been read in the current session.
#
# Uses relationships.yaml couplings to determine what docs must be read,
# checks the session reads file populated by track-reads.sh, and appends
# structured observability entries to .claude/hook_log.jsonl.
#
# Exit codes:
#   0 - Allow (all required docs read, or nothing is gated for this file)
#   2 - Block (required docs not yet read)
#
# Bypass: SKIP_READ_GATE=1 in environment.

set -euo pipefail
extract_section_items() {
    local text="$1"
    local section_label="$2"
    printf '%s\n' "$text" | awk '
        BEGIN {capture = 0}
        $0 ~ label {capture = 1; next}
        capture && $0 ~ /^  [A-Za-z ][A-Za-z ]*:/ {capture = 0}
        capture && $0 ~ /^    - / {
            sub(/^    - /, "", $0)
            print $0
        }
        capture && $0 !~ /^    - / && $0 !~ /^$/ {
            capture = 0
        }
    ' label="$section_label"
}

build_block_context() {
    local summary="$1"
    local missing_reads
    local file_path
    local output="$summary"
    local rel_doc

    missing_reads="$(extract_section_items "$summary" "^  missing:")"
    if [[ -z "$missing_reads" ]]; then
        printf '%s' "$output"
        return 0
    fi

    output+=$'\n\nDOCUMENT CONTENTS FOR MISSING READS:'
    while IFS= read -r rel_doc; do
        [[ -z "$rel_doc" ]] && continue
        file_path="$REPO_ROOT/$rel_doc"
        output+=$'\n\n---\n'
        output+="${rel_doc}"
        output+=$'\n'
        if [[ -f "$file_path" ]]; then
            output+=$(cat "$file_path")
        else
            output+="(missing file: ${rel_doc})"
        fi
    done <<< "$missing_reads"

    printf '%s' "$output"
}

trim_for_limit() {
    local context="$1"
    local limit="$2"
    local bytes
    bytes="$(printf '%s' "$context" | wc -c | tr -d '[:space:]')"

    if ! [[ "$limit" =~ ^[0-9]+$ ]] || (( limit <= 0 )); then
        printf '%s' "$context"
        return 0
    fi
    if (( bytes <= limit )); then
        printf '%s' "$context"
        return 0
    fi

    cat <<EOF
Read the listed documents and retry:
$(extract_section_items "$context" "^  missing:")
EOF
}

format_gate_context() {
    local summary="$1"
    local include_docs="$2"
    local max_bytes="${CLAUDE_GATE_CONTEXT_MAX_BYTES:-0}"
    local context="$summary"

    if [[ "$include_docs" == "1" ]]; then
        context="$(build_block_context "$summary")"
    fi

    trim_for_limit "$context" "$max_bytes"
}

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

log_gate_decision() {
    local decision="$1"
    local reason="$2"
    local context_emitted="$3"
    local context_bytes="$4"
    if [[ ! -f "$HOOK_LOG_SCRIPT" ]]; then
        return 0
    fi
    local -a command=(
        python "$HOOK_LOG_SCRIPT" gate
        --file-path "$REL_PATH"
        --tool-name "${TOOL_NAME:-unknown}"
        --decision "$decision"
        --reason "$reason"
        --reads-file "$READS_FILE"
        --log-file "$LOG_FILE"
        --context-bytes "$context_bytes"
    )
    if [[ -n "$CHECK_CONFIG" ]]; then
        command+=(--config "$CHECK_CONFIG")
    fi
    if [[ "$context_emitted" == "1" ]]; then
        command+=(--context-emitted)
    fi
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
TOOL_NAME="$(echo "$INPUT" | jq -r '.tool_name // empty' 2>/dev/null || echo "")"
FILE_PATH="$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null || echo "")"

REPO_ROOT="$(resolve_repo_root)"
READS_FILE="$(resolve_data_path "${CLAUDE_SESSION_READS_FILE:-/tmp/.claude_session_reads}")"
LOG_FILE="$(resolve_data_path "${CLAUDE_HOOK_LOG_FILE:-.claude/hook_log.jsonl}")"
HOOK_LOG_SCRIPT="$REPO_ROOT/scripts/meta/hook_log.py"
CHECK_SCRIPT="$REPO_ROOT/scripts/check_required_reading.py"
CHECK_CONFIG="${CLAUDE_CHECK_REQUIRED_READING_CONFIG:-}"
REL_PATH="$(normalize_repo_path "$FILE_PATH")"
HOOK_EXPERIMENT_ID="${CLAUDE_HOOK_EXPERIMENT_ID:-}"
HOOK_VARIANT_ID="${CLAUDE_HOOK_VARIANT_ID:-}"
HOOK_DOWNSTREAM_RUN_ID="${CLAUDE_HOOK_DOWNSTREAM_RUN_ID:-}"

if [[ "$TOOL_NAME" != "Edit" && "$TOOL_NAME" != "Write" ]]; then
    log_gate_decision "skip" "non-edit tool" "0" "0"
    exit 0
fi

if [[ -z "$FILE_PATH" ]]; then
    log_gate_decision "skip" "missing file path" "0" "0"
    exit 0
fi

if [[ "${SKIP_READ_GATE:-}" == "1" ]]; then
    log_gate_decision "skip" "SKIP_READ_GATE=1" "0" "0"
    exit 0
fi

if [[ ! -f "$CHECK_SCRIPT" ]]; then
    log_gate_decision "skip" "missing check_required_reading.py" "0" "0"
    exit 0
fi

set +e
if [[ -n "$CHECK_CONFIG" ]]; then
    RESULT="$(cd "$REPO_ROOT" && python "$CHECK_SCRIPT" "$REL_PATH" --reads-file "$READS_FILE" --config "$CHECK_CONFIG" 2>/dev/null)"
else
    RESULT="$(cd "$REPO_ROOT" && python "$CHECK_SCRIPT" "$REL_PATH" --reads-file "$READS_FILE" 2>/dev/null)"
fi
CHECK_EXIT=$?
set -e

if [[ $CHECK_EXIT -ne 0 ]]; then
    CONTEXT_PAYLOAD="$(format_gate_context "$RESULT" "1")"
    CONTEXT_BYTES="$(printf '%s' "$CONTEXT_PAYLOAD" | wc -c | tr -d '[:space:]')"
    RESULT_ESCAPED="$(echo "$CONTEXT_PAYLOAD" | jq -Rs .)"
    log_gate_decision "block" "required reading missing" "1" "$CONTEXT_BYTES"

    cat << EOF
{
  "decision": "block",
  "reason": $RESULT_ESCAPED,
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "additionalContext": $RESULT_ESCAPED
  }
}
EOF
    exit 2
fi

if [[ -n "$RESULT" ]]; then
    WARNINGS="$(extract_section_items "$RESULT" "^  scope warnings:")"
    if [[ -n "$WARNINGS" ]]; then
        CONTEXT_BYTES="$(printf '%s' "$RESULT" | wc -c | tr -d '[:space:]')"
        RESULT_ESCAPED="$(echo "$RESULT" | jq -Rs .)"
        log_gate_decision "allow" "scope warnings present" "1" "$CONTEXT_BYTES"
        cat << EOF
{
  "reason": $RESULT_ESCAPED,
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "additionalContext": $RESULT_ESCAPED
  }
}
EOF
        exit 0
    fi

    CONTEXT_BYTES="$(printf '%s' "$RESULT" | wc -c | tr -d '[:space:]')"
    log_gate_decision "allow" "required reading satisfied" "1" "$CONTEXT_BYTES"
    RESULT_ESCAPED="$(echo "$RESULT" | jq -Rs .)"
    cat << EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "additionalContext": $RESULT_ESCAPED
  }
}
EOF
else
    log_gate_decision "allow" "required reading satisfied" "0" "0"
fi

exit 0
