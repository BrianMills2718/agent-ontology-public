from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_VIEWER_HTML = REPO_ROOT / "spec-viewer.html"


def _viewer_text() -> str:
    return SPEC_VIEWER_HTML.read_text(encoding="utf-8")


def test_file_snapshot_controls_are_present() -> None:
    text = _viewer_text()
    assert 'id="live-file-refresh-btn"' in text
    assert 'id="live-file-load-snapshot-btn"' in text
    assert 'id="live-file-load-workspace-btn"' in text
    assert 'id="live-file-load-status"' in text
    assert 'id="live-file-prev-version-btn"' in text
    assert 'id="live-file-next-version-btn"' in text
    assert 'id="live-file-load-more-btn"' in text
    assert 'id="live-file-wrap-toggle"' in text
    assert 'id="live-file-mismatch"' in text


def test_live_run_defaults_use_unrestricted_profile() -> None:
    text = _viewer_text()
    assert '<option value="live" selected>live</option>' in text
    assert '<option value="external_enabled" selected>external_enabled</option>' in text
    assert "payloadOverride.mock_mode || 'live'" in text
    assert "payloadOverride.capability_profile || 'external_enabled'" in text
    assert 'id="live-advanced-toggle"' in text
    assert "function setLiveAdvancedMode(enabled)" in text
    assert "liveAdvancedMode = false;" in text
    assert "ui: simple" in text
    assert "document.body.classList.toggle('live-advanced-mode', liveAdvancedMode);" in text


def test_sink_interaction_contract_supports_highlight_and_filter_modes() -> None:
    text = _viewer_text()
    assert "let smLiveSinkSelectionMode = 'highlight';" in text
    assert "function smSelectSink(evOrKind, maybeKind)" in text
    assert "evOrKind.shiftKey" in text
    assert "smLiveSinkSelectionMode = nextMode;" in text
    assert "sinkFilterActive()" in text
    assert "smLiveSinkSelectionMode === 'filter'" in text


def test_global_cursor_contract_functions_exist() -> None:
    text = _viewer_text()
    assert "let liveCursorEventIndex = null;" in text
    assert "function setLiveCursorEventIndex(value" in text
    assert "function syncLiveSelectionsFromCursor" in text
    assert "function findClosestRowByEventIndex(rows, targetEventIndex)" in text
    assert 'id="live-cursor-status"' in text
    assert "function renderLiveCursorStatus()" in text
    assert "selected_event_index:" in text


def test_event_rows_capture_event_index_for_cross_pane_sync() -> None:
    text = _viewer_text()
    assert "event_index: eventIndex" in text
    assert "event_index: selected.event_index" in text
    assert "setLiveCursorEventIndex(row.event_index" in text


def test_canonical_live_events_store_and_fallback_hooks_exist() -> None:
    text = _viewer_text()
    assert "let liveEvents = [];" in text
    assert "let liveEventSeq = 0;" in text
    assert "let liveProjectionCache =" in text
    assert "function recomputeLiveProjectionCache()" in text
    assert "function getLiveProjectionRows(kind)" in text
    assert "function appendCanonicalLiveEvent(type, payload" in text
    assert "function maybeAutoFallbackRun(trigger = 'runtime_error')" in text
    assert "maybeAutoFallbackRun('completed_event')" in text


def test_trace_pane_is_primary_with_internal_modes() -> None:
    text = _viewer_text()
    assert 'data-live-pane="trace"' in text
    assert 'data-live-trace-mode="log"' in text
    assert 'live-advanced-only active" data-live-trace-mode="log"' in text
    assert 'data-live-trace-mode="payload"' in text
    assert 'data-live-trace-mode="diffs"' in text
    assert 'data-live-trace-mode="store_diffs"' in text
    assert 'data-live-trace-mode="file_diffs"' in text
    assert 'data-live-trace-mode="store_file_split"' in text
    assert 'data-live-trace-mode="flow"' in text
    assert 'data-live-trace-mode="uml"' in text
    assert "function setLiveTraceMode(mode" in text
    assert 'data-live-pane="diffs"' not in text
    assert 'data-live-pane="store_diffs"' not in text
    assert 'data-live-pane="file_diffs"' not in text
    assert 'data-live-pane="store_file_split"' not in text


def test_memory_flow_store_graph_contract_exists() -> None:
    text = _viewer_text()
    assert 'id="live-store-flow-map"' in text
    assert "data-live-preset=\"execution\"" in text
    assert "id=\"live-replay-play-btn\"" in text
    assert "function renderLiveStoreFlowMap()" in text
    assert "function buildLiveStoreActivitySnapshot()" in text
    assert "function findGraphStoreNodeByLiveKey(storeName)" in text
    assert "function syncGraphSelectionFromLiveStore(storeName)" in text
    assert "function setLiveFlowHoverStore(storeName)" in text
    assert "function toggleLiveCursorReplay()" in text
    assert "function ensureLiveSnapshotAtCursor({ quiet = true } = {})" in text
    assert "source: snapshot@cursor" in text
    assert "function emitInferredStoreReadsForProcess(processId, statePayload)" in text
    assert "store_read" in text
    assert "function collectStoreVectorDatasets(value, path = '$'" in text
    assert "function buildSidebarVectorMapModel(dataset, snippets = [], options = {})" in text
    assert "function renderSidebarVectorMap(model)" in text
    assert "id=\"sb-vector-dataset-select\"" in text
    assert "id=\"sb-vector-projection-select\"" in text
    assert "id=\"sb-vector-cluster-select\"" in text
    assert "function bindSidebarVectorMapControls()" in text
    assert "function renderSidebarVectorPointDetail(model, pointIndex = null" in text
    assert "function findLiveMemorySnippetPath(storeKey, snippet)" in text
    assert "function storeKeyFromMemoryPath(path)" in text


def test_verify_retry_causal_hints_are_projected_into_flow_and_sequence() -> None:
    text = _viewer_text()
    assert "function deriveVerifyRetryCausalHint(type, payload, process, detail)" in text
    assert "verify_progress" in text
    assert "checkpoint_state" in text
    assert "resume_checkpoint" in text
    assert "summaryWithCausality" in text


def test_related_run_clickthrough_contract_exists() -> None:
    text = _viewer_text()
    assert 'id="live-trace-payload-related-runs"' in text
    assert "function extractRelatedRunIdsFromRow(row)" in text
    assert "function loadLiveRunTimeline(runId, options = {})" in text
    assert "function renderLiveTracePayloadRelatedRuns(row)" in text
    assert "/api/live/events/" in text


def test_routing_telemetry_strip_contract_exists() -> None:
    text = _viewer_text()
    assert 'id="live-routing-telemetry"' in text
    assert "function resolveLiveRoutingTelemetrySnapshot()" in text
    assert "function renderLiveRoutingTelemetry()" in text
    assert "semantic_fallback_count" in text
    assert "routing_override_reason" in text
    assert "executed_interface" in text


def test_workspace_memory_explorer_contract_exists() -> None:
    text = _viewer_text()
    assert 'class="live-workspace-layout"' in text
    assert 'class="live-workspace-sidebar"' in text
    assert 'class="live-workspace-main"' in text
    assert "live-workspace-sidebar-subtitle" in text
    assert 'id="live-explorer-controls"' in text
    assert 'id="live-explorer-toggle-btn"' in text
    assert 'id="live-explorer-header-btn"' in text
    assert 'id="live-show-explorer-btn"' in text
    assert 'id="live-explorer-hidden-banner"' in text
    assert 'id="live-row-files"' in text
    assert 'id="live-row-constraints"' in text
    assert 'id="live-row-mode"' in text
    assert 'id="live-row-model"' in text
    assert 'id="live-row-runtime"' in text
    assert 'id="live-row-actions"' in text
    assert 'id="live-options-wrap"' in text
    assert 'id="live-trace-toolbar"' in text
    assert 'id="live-advanced-controls"' in text
    assert 'id="live-advanced-controls-body"' in text
    assert "Advanced Run Controls" in text
    assert 'id="live-panel-explorer-toggle"' in text
    assert 'id="live-panel-trace-detail-toggle"' in text
    assert 'id="live-panel-output-sticky-toggle"' in text
    assert 'id="live-memory-explorer"' in text
    assert 'id="live-memory-tree"' in text
    assert 'id="live-memory-detail"' in text
    assert "function setLiveExplorerDockCollapsed(collapsed)" in text
    assert "function restoreLiveExplorerLayout()" in text
    assert "LIVE_EXPLORER_LAYOUT_KEY" in text
    assert "LIVE_PANEL_PREFS_KEY" in text
    assert "function applyLivePanelPrefs(" in text
    assert "function restoreLivePanelPrefs()" in text
    assert "function mountLiveControlPlaneInExplorer()" in text
    assert "'live-trace-toolbar'" in text
    assert "'live-tabs'" in text
    assert "const advancedBody = document.getElementById('live-advanced-controls-body');" in text
    assert "liveTraceDetailVisible = false;" in text
    assert "function buildLiveMemoryExplorerModel()" in text
    assert "function renderLiveMemoryExplorer()" in text
    assert "function setLiveMemorySelected(path" in text
    assert 'id="live-quality-filter-btn"' in text
    assert "live-advanced-only" in text
    assert "setLiveTraceMode('payload')" in text


def test_quality_gate_filter_and_status_contract_exists() -> None:
    text = _viewer_text()
    assert "function liveQualitySnapshotFromObject(source)" in text
    assert "function resolveLiveQualityGateSnapshot()" in text
    assert "function setLiveQualityFilterOnly(enabled)" in text
    assert "quality_filter:" in text
    assert "quality_reason:" in text
    assert "quality_gate_failed" in text


def test_live_chat_session_contract_exists() -> None:
    text = _viewer_text()
    assert 'id="live-chat-wrap"' in text
    assert 'id="live-chat-transcript"' in text
    assert 'id="live-chat-reset-btn"' in text
    assert 'id="live-chat-session-select"' in text
    assert "LIVE_CHAT_STATE_KEY" in text
    assert "function createLiveChatSessionId()" in text
    assert "function saveLiveChatSessions()" in text
    assert "function restoreLiveChatSessions()" in text
    assert "function selectLiveChatSession(sessionId)" in text
    assert "function renderLiveChatTranscript()" in text
    assert "function pushLiveChatTurn(role, content, meta = {})" in text
    assert "function resetLiveChatSession()" in text
    assert "conversation_history: conversationHistory" in text
    assert "chat_session_id: liveChatSessionId" in text
    assert "messages: conversationHistory" in text
