# Agent Ontology Spec Format — Version 1.0

**Status:** Draft
**Date:** 2026-02-09
**Ontology version:** 0.2

This document is the standalone specification for the Agent Ontology YAML agent spec format. A developer who has never seen the Agent Ontology repository should be able to write a valid spec from this document alone.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Spec Structure](#2-spec-structure)
3. [Entity Types](#3-entity-types)
4. [Process Types](#4-process-types)
5. [Edge Types](#5-edge-types)
6. [Schemas](#6-schemas)
7. [Type System](#7-type-system)
8. [Validation Rules](#8-validation-rules)
9. [Conventions](#9-conventions)
10. [Complete Example](#10-complete-example)

---

## 1. Overview

An Agent Ontology spec is a YAML file that describes an AI agent architecture as a typed graph. The graph has three kinds of nodes:

- **Entities** — things that exist (agents, stores, tools, humans, channels, teams, conversations, configs)
- **Processes** — things that happen (steps, gates, checkpoints, spawns, protocols, policies, error handlers)
- **Schemas** — data shapes that flow between nodes

Nodes are connected by **edges** (flow, invoke, loop, branch, read, write, publish, subscribe, handoff, error, modify, observe).

The spec is declarative: it describes *what* the architecture looks like, not *how* to execute it. Code generators (instantiators) read the spec and produce runnable code.

### Design Principles

1. **Reified hypergraph substrate.** Any relationship can be promoted to a first-class node. Binary relationships use `from`/`to` shorthand (edges). N-ary relationships become process nodes (e.g., `protocol` with participants and roles).
2. **Ontology-constrained.** The spec format defines a fixed set of node types, edge types, and required fields. This makes specs machine-analyzable.
3. **Sufficient for rendering and code generation.** A valid spec contains enough information to (a) render an architecture diagram and (b) generate a runnable agent.

---

## 2. Spec Structure

A spec file is a YAML document with the following top-level keys:

```yaml
name: "My Agent"                    # REQUIRED. Human-readable name.
version: "1.0"                      # REQUIRED. Spec version string.
description: "What this agent does" # Optional. Free-text description.
entry_point: first_step             # Optional. ID of the first process to execute.
                                    # If omitted, the first process with no incoming
                                    # flow/loop edges is used.

entities:   []   # REQUIRED. List of entity objects.
processes:  []   # REQUIRED. List of process objects.
edges:      []   # REQUIRED. List of edge objects.
schemas:    []   # Optional. List of schema definitions.

metadata:   {}   # Optional. Arbitrary key-value metadata.

# Optional: spec-level state configuration
state:
  schema: StateName              # Named schema defining the full state shape
  channels:                      # Typed channels with reducers
    - { name: field_name, type: "list<Item>", reducer: append }
  initial:                       # Default initial state values
    field_name: []

# Optional: checkpointing configuration
checkpointing:
  enabled: true
  strategy: every_step           # every_step | every_gate | on_error | manual
  storage: memory                # memory | file | database
  time_travel: false             # Whether forking from past checkpoints is supported
```

### Identifiers

Every entity and process has a unique `id` (string). IDs must be unique across all entities and processes within a spec. IDs are used as reference targets in edges and other fields.

### References

Several fields contain **references** to other nodes or schemas:

| Reference type | Points to | Example |
|---------------|-----------|---------|
| `node_ref` | Any entity or process ID | `from: think_step` |
| `schema_ref` | A schema name | `data_out: ReActStep` |
| `agent_ref` | An agent entity ID | `manager: lead_agent` |
| `tool_ref` | A tool entity ID | `tools: [search_tool]` |
| `spec_ref` | An external spec file | `subgraph: other_agent.yaml` |

---

## 3. Entity Types

Entities are the persistent "nouns" of the architecture. Each entity object has `id`, `type`, and `label` (all required), plus type-specific fields.

### 3.1 `agent`

An LLM-based reasoning unit.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | Unique identifier |
| `type` | yes | `"agent"` | |
| `label` | yes | string | Human-readable name |
| `model` | yes | string | LLM model identifier (e.g., `gemini-3-flash-preview`, `claude-opus-4-6`, `gpt-4o`) |
| `system_prompt` | no | string | System prompt text (may use YAML `\|` for multiline) |
| `tools` | no | list\<tool_ref\> | Tool entities this agent can invoke |
| `input_schema` | no | schema_ref | Expected input shape |
| `output_schema` | no | schema_ref | Expected output shape |
| `config` | no | object | LLM config: `temperature` (float 0-2), `max_tokens` (int), `thinking` (none\|low\|high\|extended), `stop` (list\<string\>) |
| `subgraph` | no | spec_ref | If this agent is itself a full spec (recursive composition) |

**Example:**
```yaml
- id: reasoning_agent
  type: agent
  label: "Reasoning Agent"
  model: gemini-3-flash-preview
  system_prompt: |
    You are a ReAct agent. Alternate between thinking and acting.
  input_schema: ReActInput
  output_schema: ReActStep
```

### 3.2 `store`

A persistence layer — database, vector store, queue, etc.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"store"` | |
| `label` | yes | string | |
| `store_type` | yes | enum | `vector` \| `file` \| `kv` \| `queue` \| `relational` \| `blackboard` |
| `schema` | no | schema_ref | Data schema for stored items |
| `retention` | no | enum | `ephemeral` \| `session` \| `persistent` (default: `persistent`) |
| `access` | no | enum | `read` \| `write` \| `readwrite` (default: `readwrite`) |
| `config` | no | object | Store-specific configuration |

**Example:**
```yaml
- id: trajectory_store
  type: store
  label: "Trajectory"
  store_type: queue
  schema: TrajectoryEntry
  retention: session
```

### 3.3 `tool`

An external capability an agent can invoke.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"tool"` | |
| `label` | yes | string | |
| `tool_type` | yes | enum | `api` \| `function` \| `browser` \| `shell` \| `mcp` \| `composite` |
| `description` | no | string | What this tool does |
| `input_schema` | no | schema_ref | |
| `output_schema` | no | schema_ref | |
| `side_effects` | no | list\<string\> | What external state this tool modifies |
| `idempotent` | no | boolean | Default: `false` |
| `auth_required` | no | boolean | Default: `false` |

**Example:**
```yaml
- id: search_tool
  type: tool
  label: "Search"
  tool_type: api
  description: "Web search for factual information"
  input_schema: ToolInput
  output_schema: ToolOutput
```

### 3.4 `human`

A human participant in the system.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"human"` | |
| `label` | yes | string | |
| `role` | no | enum | `user` \| `reviewer` \| `admin` \| `operator` |

**Example:**
```yaml
- id: user
  type: human
  label: "User"
  role: user
```

### 3.5 `config`

A configuration/environment object.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"config"` | |
| `label` | yes | string | |
| `values` | no | object | Key-value configuration |

### 3.6 `channel`

A named pub/sub communication channel.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"channel"` | |
| `label` | yes | string | |
| `channel_type` | yes | enum | `topic` \| `queue` \| `broadcast` \| `request_reply` |
| `message_schema` | no | schema_ref | Schema of messages on this channel |
| `retention` | no | enum | `none` \| `last` \| `all` \| `windowed` (default: `all`) |
| `reducer` | no | enum | `append` \| `replace` \| `merge` \| `custom` (default: `append`). How concurrent writes merge. |
| `buffer_size` | no | integer or `"unbounded"` | Default: `unbounded` |

**Example:**
```yaml
- id: requirements_channel
  type: channel
  label: "Requirements Channel"
  channel_type: topic
  message_schema: PRD
  retention: all
```

### 3.7 `team`

A group of agents with a collective execution strategy.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"team"` | |
| `label` | yes | string | |
| `members` | yes | list\<agent_ref\> | Agent entities in this team |
| `strategy` | yes | enum | `sequential` \| `hierarchical` \| `consensus` \| `round_robin` \| `dynamic` |
| `manager` | no | agent_ref | Manager agent (for hierarchical strategy) |
| `delegation` | no | boolean | Whether members can delegate to each other (default: `false`) |
| `speaker_selection` | no | enum | `round_robin` \| `llm_based` \| `priority` \| `random` \| `custom` |
| `max_rounds` | no | integer | |
| `termination` | no | termination_condition or string | When the team's work is complete |

**Example:**
```yaml
- id: dev_team
  type: team
  label: "Development Team"
  members: [product_manager, architect, engineer, qa_agent]
  strategy: sequential
```

### 3.8 `conversation`

A multi-turn dialogue with structured history.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"conversation"` | |
| `label` | yes | string | |
| `participants` | no | list\<node_ref\> | Agents in this conversation |
| `history_schema` | no | schema_ref | Schema for conversation turns |
| `max_turns` | no | integer | |
| `persistence` | no | enum | `ephemeral` \| `session` \| `persistent` (default: `session`) |
| `nesting` | no | boolean | Whether sub-conversations are allowed (default: `false`) |

---

## 4. Process Types

Processes are the "verbs" — things that happen. They are the reified hyperedges of the graph. Each process has `id`, `type`, and `label` (all required).

### 4.1 `step`

A deterministic orchestration point that transforms data, coordinates entities, or performs logic.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"step"` | |
| `label` | yes | string | |
| `description` | no | string | What this step does |
| `logic` | no | string | Inline code (Python). Executed during this step. Has access to `state.data` dict. |
| `data_in` | no | schema_ref | Input schema (for building LLM prompts when this step invokes an agent) |
| `data_out` | no | schema_ref | Output schema (for parsing LLM responses) |
| `timeout` | no | duration | Max execution time |
| `on_error` | no | enum | `fail` \| `skip` \| `retry` \| `fallback` (default: `fail`) |

**Logic blocks** are inline Python code that runs during the step's execution. They have access to `state.data` (a dict containing all runtime state). Logic blocks should be kept minimal — data reshaping, counter management, debug prints. Heavy computation belongs in LLM calls or tools.

**Example:**
```yaml
- id: receive_query
  type: step
  label: "Receive Query"
  data_out: ReActInput
  logic: |
    state.data["trajectory"] = []
    state.data["step_count"] = 0
    state.data["max_steps"] = 10
```

### 4.2 `gate`

A decision/branch point that evaluates a condition and routes to different paths.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"gate"` | |
| `label` | yes | string | |
| `condition` | yes | string | The condition being evaluated (human-readable) |
| `branches` | yes | list\<branch\> | Each branch has `condition` (string) and `target` (node_ref) |
| `default` | no | node_ref | Fallback if no branch matches |
| `logic` | no | string | Optional inline code (e.g., to increment counters before branching) |

A gate must have at least 2 branches. Branch conditions are human-readable strings evaluated against `state.data` keys.

**Example:**
```yaml
- id: check_quality
  type: gate
  label: "Quality >= threshold?"
  condition: "quality_score >= quality_threshold"
  branches:
    - condition: "score >= 7"
      target: finalize
    - condition: "score < 7"
      target: check_rounds
```

### 4.3 `checkpoint`

A human-in-the-loop pause point.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"checkpoint"` | |
| `label` | yes | string | |
| `prompt` | yes | string | What to show the human |
| `timeout` | no | duration | How long to wait |
| `default_action` | no | enum | `approve` \| `deny` \| `skip`. What happens on timeout. |
| `options` | no | list\<string\> | Choices presented to human |

**Example:**
```yaml
- id: request_fixes
  type: checkpoint
  label: "Request Fixes from User"
  prompt: "Please address the issues and push new commits."
  options: ["fixes pushed"]
  timeout: 86400
  default_action: "skip"
```

### 4.4 `spawn`

Dynamic instantiation — creates agent(s) at runtime from a template.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"spawn"` | |
| `label` | yes | string | |
| `template` | yes | agent_ref, spec_ref, or `"self"` | What to instantiate. `"self"` = recursive. |
| `cardinality` | no | integer or `"dynamic"` | How many to create (default: 1) |
| `determined_by` | no | node_ref | Which node decides cardinality (if dynamic) |
| `aggregation` | no | enum | `collect` \| `merge` \| `vote` \| `first` \| `race` |
| `recursive` | no | boolean | Whether spawned agents can themselves spawn (default: `false`) |
| `max_depth` | no | integer or `"unbounded"` | Maximum recursion depth |

**Example:**
```yaml
- id: recurse_decomposition
  type: spawn
  label: "Recursive Decomposition"
  template: meta_agent
  cardinality: 1
  recursive: true
  max_depth: 3
  aggregation: collect
```

### 4.5 `protocol`

A multi-party interaction pattern (negotiation, consensus, auction).

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"protocol"` | |
| `label` | yes | string | |
| `participants` | yes | list\<participant\> | Each has `entity` (node_ref) and `role` (string). Min 2. |
| `termination` | yes | termination_condition or string | When the protocol ends |
| `rules` | no | list\<string\> | Protocol rules in order |
| `state` | no | schema_ref | Shared state schema |
| `max_rounds` | no | integer | |

**Example:**
```yaml
- id: negotiate_conflicts
  type: protocol
  label: "Conflict Negotiation"
  participants:
    - entity: meta_agent
      role: mediator
    - entity: worker_agent
      role: disputant
    - entity: integrator_agent
      role: arbitrator
  termination: "consensus reached or mediator decides after 2 rounds"
  max_rounds: 2
```

### 4.6 `policy`

A cross-cutting rule that modifies behavior of other nodes (guardrails, rate limits, safety filters).

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"policy"` | |
| `label` | yes | string | |
| `targets` | yes | list\<node_ref\> | What this policy applies to |
| `effect` | yes | enum | `block` \| `warn` \| `modify` \| `log` \| `retry` |
| `condition` | no | string | When the policy activates |
| `rules` | no | list\<string\> | |
| `enforcement` | no | enum | `strict` \| `advisory` (default: `strict`) |

### 4.7 `error_handler`

Structured error handling with retry, fallback, and cleanup.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `id` | yes | string | |
| `type` | yes | `"error_handler"` | |
| `label` | yes | string | |
| `scope` | yes | list\<node_ref\> | Processes covered by this handler |
| `on_error` | yes | node_ref | Where to route on error (catch block) |
| `retry` | no | object | `max_retries` (int, default 3), `backoff` (none\|linear\|exponential), `initial_delay_ms` (int), `max_delay_ms` (int), `retryable_errors` (list\<string\>) |
| `fallback` | no | node_ref | Alternative path when retries exhausted |
| `on_finally` | no | node_ref | Always-run cleanup |
| `error_schema` | no | schema_ref | Schema for error data passed to handler |
| `timeout` | no | duration | Max time for the entire scope |

---

## 5. Edge Types

Edges are binary connections between nodes. Each edge object has `type`, `from`, and `to` (all required), plus type-specific fields.

### 5.1 `flow`

Sequential control or data flow from A to B.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"flow"` | |
| `from` | yes | node_ref | |
| `to` | yes | node_ref | |
| `label` | no | string | What flows (shown on arrow) |
| `data` | no | schema_ref | Shape of data on this edge |

**Example:**
```yaml
- type: flow
  from: receive_query
  to: think_or_act
  label: "Start reasoning"
```

### 5.2 `invoke`

A process calls an entity and expects a response. This is the core pattern for LLM calls and tool use.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"invoke"` | |
| `from` | yes | node_ref | The caller (must be a process: step, gate, or spawn) |
| `to` | yes | node_ref | The callee (must be an entity: agent or tool) |
| `label` | no | string | |
| `input` | no | schema_ref | Data sent to callee |
| `output` | no | schema_ref | Data returned from callee |
| `return_to` | no | node_ref | Where the result goes (defaults to `from`) |
| `async` | no | boolean | Default: `false` |
| `retry` | no | object | `max_retries` (int), `backoff` (enum), `initial_delay_ms` (int), `retryable_errors` (list) |
| `timeout` | no | duration | Max time for this invocation |

**Example:**
```yaml
- type: invoke
  from: think_or_act
  to: reasoning_agent
  label: "Get next step"
  input: ReActInput
  output: ReActStep
```

### 5.3 `loop`

Returns execution to an earlier node (backward edge).

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"loop"` | |
| `from` | yes | node_ref | |
| `to` | yes | node_ref | Must be earlier in the flow |
| `label` | no | string | |
| `condition` | no | string | When to loop (vs. exit) |
| `max_iterations` | no | integer | |

### 5.4 `branch`

Conditional forward path from a gate to a target.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"branch"` | |
| `from` | yes | node_ref | Must be a gate |
| `to` | yes | node_ref | |
| `condition` | yes | string | When this branch is taken |
| `label` | no | string | |
| `data` | no | schema_ref | |
| `priority` | no | integer | Evaluation order |

### 5.5 `read`

A process reads from a store.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"read"` | |
| `from` | yes | node_ref | The reader (step or agent) |
| `to` | yes | node_ref | The store |
| `label` | no | string | |
| `query` | no | schema_ref | Query/filter schema |
| `query_key` | no | string | State key containing the query value for dynamic lookups (e.g., `rewritten_query` → `state.data["rewritten_query"]` passed to store read) |
| `data` | no | schema_ref | Schema of returned data |

### 5.6 `write`

A process writes to a store.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"write"` | |
| `from` | yes | node_ref | The writer (step or agent) |
| `to` | yes | node_ref | The store |
| `label` | no | string | |
| `data` | no | schema_ref | What gets written |

### 5.7 `publish`

An entity publishes messages to a channel.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"publish"` | |
| `from` | yes | node_ref | The publisher (agent or step) |
| `to` | yes | node_ref | The target channel |
| `label` | no | string | |
| `filter` | no | string | Condition for publishing |
| `data` | no | schema_ref | Schema of published messages |

### 5.8 `subscribe`

An entity subscribes to messages from a channel.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"subscribe"` | |
| `from` | yes | node_ref | The source channel |
| `to` | yes | node_ref | The subscriber (agent or step) |
| `label` | no | string | |
| `filter` | no | string | Subscription filter expression |
| `activates` | no | boolean | Whether message arrival triggers the subscriber (default: `true`) |
| `data` | no | schema_ref | Schema of received messages |

### 5.9 `handoff`

Agent-to-agent control transfer.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"handoff"` | |
| `from` | yes | node_ref | The handing-off agent |
| `to` | yes | node_ref | The receiving agent |
| `label` | no | string | |
| `condition` | no | string | When to hand off |
| `context` | no | enum | `full` \| `summary` \| `none` (default: `full`). How much conversation context transfers. |
| `resumable` | no | boolean | Whether control can return (default: `false`) |

### 5.10 `error`

Error flow routing to an error handler.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"error"` | |
| `from` | yes | node_ref | The process that can fail |
| `to` | yes | node_ref | The error handler or recovery step |
| `label` | no | string | |
| `error_types` | no | list\<string\> | Which error types trigger this edge |
| `data` | no | schema_ref | Error data schema |

### 5.11 `modify`

A policy or config changes the behavior of a target.

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"modify"` | |
| `from` | yes | node_ref | Policy or config |
| `to` | yes | node_ref | The modified node |
| `label` | no | string | |
| `effect` | no | string | What changes |

### 5.12 `observe`

A node monitors another without affecting it (logging, tracing, meta-cognition).

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | `"observe"` | |
| `from` | yes | node_ref | The observer |
| `to` | yes | node_ref | The observed |
| `label` | no | string | |

---

## 6. Schemas

Schemas define the shape of data flowing through the system. They are referenced by `schema_ref` fields on entities, processes, and edges.

```yaml
schemas:
  - name: ReActStep              # REQUIRED. Must be unique within the spec.
    description: "One step"      # Optional.
    fields:                      # REQUIRED. List of field definitions.
      - name: type               # REQUIRED.
        type: "enum[thought, action, answer]"  # REQUIRED.
        description: "Step type" # Optional.
      - name: thought
        type: string
      - name: max_tasks
        type: integer
        default: 5               # Optional. Default value for this field.
```

Each field has:

| Property | Required | Description |
|----------|----------|-------------|
| `name` | yes | Field name |
| `type` | yes | Data type (see table below) |
| `description` | no | Human-readable description |
| `default` | no | Default value if not provided at runtime |

### Field Types

| Type | Description | Examples |
|------|-------------|---------|
| `string` | Text | `"hello"` |
| `integer` | Whole number | `42` |
| `float` | Decimal number | `0.95` |
| `boolean` | True/false | `true` |
| `object` | Arbitrary nested object | `{}` |
| `list<T>` | Ordered list of type T | `list<string>`, `list<Issue>` |
| `enum[a, b, c]` | One of the listed values | `enum[thought, action, answer]` |
| `SchemaName` | Reference to another schema | `SecurityReview`, `list<Issue>` |

Schemas can reference other schemas in field types, enabling nested data structures:

```yaml
schemas:
  - name: SynthesizerInput
    fields:
      - { name: security_review, type: SecurityReview }
      - { name: style_review, type: StyleReview }
      - { name: logic_review, type: LogicReview }
```

---

## 7. Type System

### 7.1 Termination Conditions

Used by `protocol`, `team`, and anywhere loop/protocol termination is needed. Can be a simple string or a structured composable condition:

**Simple (string):**
```yaml
termination: "consensus reached or 3 rounds"
```

**Structured:**
```yaml
termination:
  operator: or
  conditions:
    - { max_turns: { count: 10 } }
    - { text_match: { pattern: "CONSENSUS_REACHED", in_field: "last_message" } }
```

Variants: `simple` (condition string), `max_turns` (count), `max_time` (duration), `text_match` (pattern + in_field), `composite` (operator: and|or|not + sub-conditions).

### 7.2 Duration

Duration fields accept integer seconds or string durations:

```yaml
timeout: 86400        # seconds
timeout: "5m"         # string format
```

---

## 8. Validation Rules

A valid spec must satisfy these rules. Rules marked `error` must be fixed; `warning` is informational.

### Errors (must fix)

1. Every spec must have at least one `agent` entity.
2. Every spec must have an `entry_point` or exactly one process with no incoming flow/loop edges.
3. Every `gate` must have at least 2 branches.
4. Every `loop` edge must target a node that appears earlier in the process list.
5. Every `schema_ref` must resolve to a defined schema name.
6. Every `spawn` template must reference a valid agent entity, spec_ref, or `"self"`.
7. Every `protocol` participant must reference a valid entity.
8. Every `error_handler` scope must reference valid process nodes.
9. Every `error_handler` `on_error` target must be a valid process node.
10. `team` members must reference valid agent entities.
11. `channel` `message_schema` must resolve to a defined schema.
12. `handoff` edges must connect two agent entities.
13. `publish` edges must go from agent/step to channel.
14. `subscribe` edges must go from channel to agent/step.
15. Composable termination conditions must use valid operators (`and`, `or`, `not`).
16. `not` operator must have exactly one sub-condition.

### Warnings (informational)

17. Every `invoke` edge should have a matching return (explicit `return_to` or implicit back to caller).
18. No orphan nodes — every node should have at least one edge.
19. Agent nodes with `tools[]` should have those tools defined as tool entities.
20. Recursive spawns should specify `max_depth` or document natural bounds.
21. Error edges should originate from processes within an `error_handler` scope.
22. Invoke edges with `retry.max_retries > 0` should specify `retryable_errors`.
23. Team `manager` (if specified) should be one of the team members.
24. Conversation participants should reference valid entity nodes.

---

## 9. Conventions

### 9.1 Entry Point

Execution starts at `entry_point`. If omitted, the first process in the `processes` list with no incoming `flow` or `loop` edges is used.

### 9.2 Process Ordering

The `processes` list defines the **definition order**. This matters for:
- Determining the entry point (first process with no incoming flow)
- Detecting backward edges (a `loop` edge targets a process earlier in the list)
- Code generation order

### 9.3 State Model

At runtime, all data lives in `state.data`, a flat dictionary. Schema fields are accessed as `state.data["field_name"]`. After an LLM invocation, the response is parsed and merged into `state.data` via `.update()`, plus stored under namespaced keys (`state.data["schema_name"]` and `state.data["process_id_result"]`).

### 9.4 Gate Conditions

Gate `condition` strings are evaluated against `state.data` keys. Supported patterns:

| Pattern | Example | Meaning |
|---------|---------|---------|
| Truthiness | `has_answer` | `state.data.get("has_answer")` is truthy |
| Equality | `type == answer` | `state.data["type"] == "answer"` |
| Comparison | `quality_score >= 7` | `state.data["quality_score"] >= 7` |
| Length | `results.length > 0` | `len(state.data["results"]) > 0` |
| Emptiness | `tool_name is not empty` | `state.data["tool_name"]` is not empty |

### 9.5 Logic Block Best Practices

- Keep logic blocks minimal: data reshaping, counter management, debug prints.
- Access state via `state.data["key"]` (flat dict, not nested paths).
- Use `state.data.get("key", default)` for safe access.
- Set `state.data["_done"] = True` to signal the agent should terminate.
- Avoid heavy computation — that belongs in LLM calls or tools.

### 9.6 Fan-Out

Multiple `flow` edges from a single process create **fan-out**: all target processes execute (sequentially in the custom backend, potentially in parallel in LangGraph). This is how the Code Reviewer runs security, style, and logic analysis in parallel.

### 9.7 Fan-In

Multiple `flow` edges converging on a single process create **fan-in**: the target waits for all incoming edges. Used after fan-out to aggregate results (e.g., synthesize_review in Code Reviewer).

### 9.8 Inline vs Explicit Branch Edges

Gate branches can be specified in two ways:

**Inline** (in the gate's `branches` list):
```yaml
- id: check_quality
  type: gate
  branches:
    - condition: "pass"
      target: deliver
    - condition: "fail"
      target: retry
```

**Explicit** (as separate `branch` edge objects):
```yaml
- type: branch
  from: check_quality
  to: deliver
  condition: "pass"
```

Both are valid. Explicit branch edges are redundant with inline branches but useful for visualization tools that read edges only. Many specs use both for clarity.

---

## 10. Complete Example

A minimal Self-Refine agent (generate → critique → refine loop):

```yaml
name: "Self Refine"
version: "1.0"
description: "Generator-critic loop"
entry_point: receive_task

entities:
  - id: generator
    type: agent
    label: "Generator"
    model: gemini-3-flash-preview
    system_prompt: "Generate high-quality output. Incorporate feedback if provided."
    output_schema: GeneratorOutput

  - id: critic
    type: agent
    label: "Critic"
    model: gemini-3-flash-preview
    system_prompt: "Evaluate output quality 1-10. Provide specific feedback."
    output_schema: CriticOutput

processes:
  - id: receive_task
    type: step
    label: "Receive Task"
    logic: |
      state.data["refinement_round"] = 0
      state.data["max_rounds"] = 3

  - id: generate
    type: step
    label: "Generate"
    data_in: GeneratorInput
    data_out: GeneratorOutput

  - id: critique
    type: step
    label: "Critique"
    data_in: CriticInput
    data_out: CriticOutput

  - id: check_quality
    type: gate
    label: "Quality OK?"
    condition: "quality_score >= 7"
    branches:
      - condition: "score >= 7"
        target: finalize
      - condition: "score < 7"
        target: refine

  - id: refine
    type: step
    label: "Refine"
    logic: |
      state.data["refinement_round"] += 1

  - id: finalize
    type: step
    label: "Done"
    logic: |
      state.data["_done"] = True

edges:
  - { type: flow, from: receive_task, to: generate }
  - { type: invoke, from: generate, to: generator, output: GeneratorOutput }
  - { type: flow, from: generate, to: critique }
  - { type: invoke, from: critique, to: critic, output: CriticOutput }
  - { type: flow, from: critique, to: check_quality }
  - { type: flow, from: refine, to: generate }
  - { type: loop, from: refine, to: generate, condition: "refinement_round < max_rounds" }

schemas:
  - name: GeneratorInput
    fields:
      - { name: task, type: string }
      - { name: specific_feedback, type: string }
      - { name: refinement_round, type: integer }

  - name: GeneratorOutput
    fields:
      - { name: output_text, type: string }
      - { name: changes_made, type: string }

  - name: CriticInput
    fields:
      - { name: task, type: string }
      - { name: output_text, type: string }

  - name: CriticOutput
    fields:
      - { name: quality_score, type: integer }
      - { name: weaknesses, type: "list<string>" }
      - { name: specific_feedback, type: string }
```

This spec validates, generates a runnable agent, and produces a Self-Refine loop that runs up to 3 refinement rounds or until quality reaches 7+.
