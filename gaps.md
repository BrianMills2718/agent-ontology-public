# Agent Ontology -- Expressiveness Gap Analysis

**Date:** 2026-02-09
**Ontology Version:** 0.2 (updated from 0.1)
**Frameworks Analyzed:** CAMEL, MetaGPT, LangGraph, OpenAI Agents SDK, CrewAI, AutoGen v0.4

> **Status:** Gaps 1-6 addressed in v0.2, Gaps 7-9 addressed in v0.2.1. Gap 10 deferred (runtime concern).
> - Gap 1 (Channels): `channel` entity + `publish`/`subscribe` edges
> - Gap 2 (Handoffs): `handoff` edge type
> - Gap 3 (Typed state): `state` field on `spec_shape` with channels/reducers
> - Gap 4 (Event-driven): `activates` flag on `subscribe` edges
> - Gap 5 (Teams): `team` entity type with strategy/delegation/speaker_selection
> - Gap 6 (Conversations): `conversation` entity type
> - Gap 7 (Checkpointing): `checkpointing` config on `spec_shape`
> - Gap 8 (Speaker selection): `speaker_selection` field on `team` entity
> - Gap 9 (Composable termination): `termination_condition` structured type

---

## 1. Current Ontology Capabilities (Summary)

The ontology (v0.1) defines a reified hypergraph model with five entity types (`agent`, `store`, `tool`, `human`, `config`), six process types (`step`, `gate`, `checkpoint`, `spawn`, `protocol`, `policy`), and seven edge types (`flow`, `invoke`, `loop`, `branch`, `read`, `write`, `modify`, `observe`). It supports schemas for typed data flowing through edges and nodes, visual rendering rules, and validation constraints.

**What it can express well today:**

- Single-agent reasoning loops (ReAct, self-refine, AutoGPT-style think-criticize-act)
- Orchestrator-worker patterns with static or dynamic spawning
- Sequential pipelines with branching and gating (plan-and-solve, code generation pipelines)
- Human-in-the-loop checkpoints
- Store-backed memory (vector, kv, file, queue, blackboard)
- Cross-cutting policies (guardrails, rate limits, logging)
- Recursive agent composition via `subgraph` and `spawn(template: 'self')`
- Fan-out/fan-in parallel execution via `spawn` with `aggregation`
- N-ary interaction protocols with roles, rules, and termination conditions

**What it struggles with or cannot express** (detailed below in Sections 3-5).

---

## 2. Framework-by-Framework Pattern Inventory

### 2.1 CAMEL (Communicative Agents)

**Core patterns:**
- **Role-playing inception prompting**: Two agents assigned complementary roles (AI User + AI Assistant) with structured inception prompts that constrain conversation dynamics
- **Autonomous turn-taking**: Agents take turns in a conversation without a central orchestrator deciding turns -- the conversation protocol itself is the control flow
- **Termination tokens**: A special message token (`<CAMEL_TASK_DONE>`) signals task completion, decided by one agent observing the conversation state
- **Conversational data generation**: The conversation history itself is a first-class artifact, not just a side effect

**Ontology friction:** The current ontology can model two agents exchanging messages via `flow` edges through `step` processes, but the *autonomous turn-taking without an orchestrator step* pattern requires wrapping each turn in a `step` + `invoke` + `gate` chain. The conversation is implicit in `state.data` rather than being a first-class typed channel.

### 2.2 MetaGPT (Software Company Simulation)

**Core patterns:**
- **Shared message pool (publish/subscribe)**: All agents publish structured messages to a global pool; agents subscribe to message types relevant to their role. No point-to-point routing -- agents pull what they need.
- **Structured intermediate artifacts**: Messages carry typed documents (PRDs, design docs, API specs in Mermaid, etc.) -- not just strings, but domain-typed intermediate work products
- **SOP-encoded workflows**: Standard Operating Procedures are first-class; the workflow topology is derived from SOPs, not manually wired
- **Role-to-SOP binding**: Each role (Product Manager, Architect, Engineer, QA) has an associated SOP that determines what it consumes and produces

**Ontology friction:** The ontology has no concept of a **message pool** or **pub/sub channel**. Agents communicate only through orchestrated `flow`/`invoke` edges. There is no way to express "agent X publishes to topic T, and agents Y and Z subscribe to T." The `store` entity can serve as a workaround (blackboard pattern), but it lacks subscription semantics -- there is no way to model reactive behavior where an agent activates upon message arrival.

### 2.3 LangGraph (Graph-Based Agent Orchestration)

**Core patterns:**
- **Typed state channels with reducers**: State is not a flat dict -- it is a set of named channels, each with a type and a reducer function (e.g., `messages: Annotated[list, add_messages]`). Channels define how concurrent writes merge.
- **Super-step execution (Pregel-inspired)**: Nodes that can run in parallel execute in the same super-step; nodes with dependencies run in subsequent super-steps. This is implicit parallelism, not explicit fan-out.
- **Checkpointing with time-travel**: Every super-step produces a checkpoint. You can fork from any checkpoint to explore alternate trajectories -- not just pause/resume.
- **Interrupt/Command flow control**: Nodes can issue `interrupt()` to pause execution and `Command(goto=...)` for dynamic routing that bypasses the static graph topology.
- **Sub-graph composition**: A node in one graph can be an entire sub-graph with its own state, creating hierarchical composition without flattening.
- **Conditional edges as functions**: Edge routing is a function of state, not just a static condition string.

**Ontology friction:** The ontology has no concept of **state channels with merge semantics**. State is implicit (carried in `state.data` in logic blocks). There is no way to express "when two parallel agents both write to the `messages` channel, append both results." Checkpointing and time-travel are not representable. The `gate` type requires a string condition, not a function reference. Sub-graph composition exists via `subgraph` on agents, but not on processes.

### 2.4 OpenAI Agents SDK (Tool Use, Handoffs, Guardrails)

**Core patterns:**
- **Agent handoff as a tool call**: One agent transfers the active conversation to another agent via a special tool call. The receiving agent inherits the full conversation history. This is not orchestrator-mediated -- the agent itself decides to hand off.
- **Conversation threads with session history**: A thread persists across runs; each run appends to the thread. History management is automatic.
- **Guardrails as input/output validators**: Guardrails are typed validators (Pydantic models or functions) that run before/after agent execution, with tripwire semantics (halt on violation).
- **Built-in tool types**: File search, code interpreter, and web search are first-class tool categories with specialized runtime behavior (sandboxed execution, vector indexing).

**Ontology friction:** **Agent-to-agent handoff** is not an edge type or process type. The current model requires an orchestrating `step` to mediate all agent-to-agent communication. There is no way to express "agent A decides mid-conversation to transfer control to agent B, including the conversation context." The `policy` process type partially covers guardrails, but lacks the input/output validator semantics and tripwire behavior. **Conversation history** as a typed, append-only channel between runs is not representable.

### 2.5 CrewAI (Role-Based Multi-Agent)

**Core patterns:**
- **Process types (sequential, hierarchical, consensual)**: The execution strategy is a first-class concept: tasks run sequentially, under a manager, or via consensus voting. This is metadata on the *crew*, not encoded in the graph topology.
- **Agent delegation**: An agent can dynamically delegate a sub-task to another agent during execution, without pre-wired edges. The delegating agent chooses the delegate based on capability matching.
- **Flows as event-driven orchestration**: CrewAI Flows use `@listen()` and `@router()` decorators to create event-driven state machines where methods trigger on the completion of other methods.
- **Structured state with Pydantic models**: Flow state is a typed Pydantic model, not a free-form dict.
- **Crew as a composable unit**: A Crew (group of agents + tasks + process type) is itself a callable unit within a Flow, enabling hierarchical composition.

**Ontology friction:** The ontology lacks a **crew/team entity** with an associated execution strategy. The crew spec (`specs/crew.yaml`) must manually encode sequential/hierarchical behavior in the graph topology rather than declaring it. **Dynamic delegation** (agent A decides at runtime to ask agent B for help) cannot be expressed because all inter-agent communication requires pre-wired edges. **Event-driven activation** (agent runs when a specific condition on state becomes true) is not a concept -- all activation is via explicit flow/invoke edges.

### 2.6 AutoGen v0.4 (Conversational Multi-Agent)

**Core patterns:**
- **Actor model with typed messages**: Each agent is an actor with a mailbox. Agents send and receive typed messages. The runtime routes messages based on subscriptions and agent IDs.
- **Topic-based pub/sub**: Agents subscribe to topics. Publishing a message to a topic delivers it to all subscribers. This decouples producers from consumers.
- **GroupChat with speaker selection**: A GroupChatManager selects the next speaker using configurable strategies (round-robin, LLM-based, custom function). The selection logic is a first-class concept, not hard-coded in graph edges.
- **Distributed runtime**: Agents can run on different processes/machines. The runtime handles serialization and transport transparently.
- **Termination conditions as composable objects**: Termination is a composable condition (MaxMessages AND/OR TextMention AND/OR custom), not a single string expression.
- **Nested chat**: An agent can internally run a multi-agent conversation to resolve a sub-task before responding to the outer conversation.

**Ontology friction:** The ontology has **no message type system**. All data flows through `schema_ref` on edges, but there is no concept of typed messages with routing metadata. **Topic-based pub/sub** is absent. **Speaker selection strategies** for group conversations are not representable -- the `protocol` process type has `participants` and `rules`, but no concept of dynamic speaker selection. **Distributed deployment** concerns are entirely out of scope. **Composable termination conditions** go beyond the `condition: string` field on loops and gates.

---

## 3. Identified Expressiveness Gaps

### Gap 1: No Message/Channel Abstraction (CRITICAL)

**Affected frameworks:** MetaGPT, AutoGen, LangGraph, CAMEL

**The problem:** The ontology treats data flow as implicit -- data moves along edges via `schema_ref`, but there is no concept of a **named, typed communication channel** that multiple agents can publish to and subscribe from. All communication is point-to-point via pre-wired `flow`/`invoke` edges.

**What cannot be expressed:**
- MetaGPT's shared message pool with subscription filtering
- AutoGen's topic-based pub/sub
- LangGraph's state channels with reducer/merge semantics
- CAMEL's conversational turn-taking without an orchestrator
- Any pattern where agent activation is *reactive* (triggered by message arrival) rather than *orchestrated* (triggered by an explicit flow edge)

**Impact:** This is the single largest gap. It forces every multi-agent interaction to be mediated by explicit step nodes, making specs verbose and failing to capture the essential communication topology.

### Gap 2: No Agent Handoff / Peer-to-Peer Transfer (CRITICAL)

**Affected frameworks:** OpenAI Agents SDK, CrewAI (delegation), AutoGen

**The problem:** Agents cannot transfer control to other agents. All inter-agent communication must pass through a process node (`step`, `gate`, `spawn`). There is no edge type or mechanism for "agent A decides to hand off to agent B."

**What cannot be expressed:**
- OpenAI Swarm/Agents SDK handoff pattern
- CrewAI dynamic delegation
- Any pattern where an agent autonomously chooses its successor

**Impact:** Forces an orchestrator-centric architecture even when the real system is peer-to-peer.

### Gap 3: No Typed State / State Schema on Specs (IMPORTANT)

**Affected frameworks:** LangGraph, CrewAI Flows

**The problem:** The `spec_shape` has `schemas` for data flowing on edges, but no concept of a **global spec state schema** -- a typed object that represents the full runtime state of the system. In practice, every spec puts logic in `state.data` as an untyped dict.

**What cannot be expressed:**
- LangGraph's `TypedDict` or `Pydantic` state with named channels and reducer annotations
- CrewAI's structured state (`BaseModel`) on Flows
- Compile-time verification that a step's `data_in` will actually be present in state when it runs

**Impact:** State management is the most error-prone part of writing specs today. Without typed state, specs cannot be validated or code-generated reliably.

### Gap 4: No Event-Driven / Reactive Activation (IMPORTANT)

**Affected frameworks:** CrewAI Flows, AutoGen, MetaGPT

**The problem:** All node activation is via explicit edges. There is no way to express "this agent/step activates when condition X on state becomes true" or "when a message arrives on topic T."

**What cannot be expressed:**
- CrewAI `@listen()` decorator pattern (method runs when another method completes)
- AutoGen subscription-based activation
- MetaGPT agents that activate upon message publication
- Any event-driven or reactive agent pattern

**Impact:** Forces all control flow to be explicitly wired, even when the real system uses implicit/reactive activation.

### Gap 5: No Team / Crew Entity Type (IMPORTANT)

**Affected frameworks:** CrewAI, MetaGPT, AutoGen

**The problem:** There is no concept of a **group of agents** as a first-class entity with properties like execution strategy (`sequential`, `hierarchical`, `consensus`), membership, and delegation rules.

**What cannot be expressed:**
- CrewAI's Crew with process type metadata
- MetaGPT's role team with SOP bindings
- AutoGen's GroupChat with speaker selection strategy
- Dynamic team formation (agents joining/leaving a team at runtime)

**Impact:** Multi-agent teams must be manually encoded as flat lists of agents with the team behavior buried in step logic.

### Gap 6: No Conversation / Thread Abstraction (IMPORTANT)

**Affected frameworks:** OpenAI Agents SDK, CAMEL, AutoGen

**The problem:** There is no way to represent an ongoing conversation with history as a first-class object. Conversation history is stuffed into `state.data` as an ad-hoc list. The `store` entity with `store_type: queue` is the closest approximation, but lacks conversation semantics (turns, roles, threading).

**What cannot be expressed:**
- OpenAI Threads that persist across runs
- CAMEL's role-playing conversations as typed artifacts
- AutoGen's nested chat (a conversation within a conversation)
- Multi-turn dialogue with structured turn metadata (speaker, timestamp, role)

### Gap 7: No Checkpointing / State Snapshots / Time-Travel (NICE-TO-HAVE)

**Affected frameworks:** LangGraph

**The problem:** The ontology has no concept of state persistence between executions, checkpointing at specific points, or forking from a previous state.

**What cannot be expressed:**
- LangGraph checkpointing at every super-step
- Time-travel (forking from a past state to explore alternatives)
- Resumable execution after system failure

### Gap 8: No Speaker Selection / Turn-Taking Strategy (NICE-TO-HAVE)

**Affected frameworks:** AutoGen, CAMEL

**The problem:** The `protocol` process type has `participants` with roles and `rules`, but no formalized concept of **how the next speaker is selected** in a multi-agent conversation. Turn-taking strategies (round-robin, LLM-based, priority-based, custom function) are a key differentiator in conversational multi-agent systems.

### Gap 9: No Composable Termination Conditions (NICE-TO-HAVE)

**Affected frameworks:** AutoGen, LangGraph

**The problem:** Termination is expressed as a `condition: string` on loops, gates, and protocols. AutoGen uses composable termination objects (`MaxMessages & TextMention | CustomCondition`). The ontology cannot express boolean composition of termination criteria.

### Gap 10: No Deployment / Runtime Topology (NICE-TO-HAVE)

**Affected frameworks:** AutoGen v0.4

**The problem:** AutoGen supports distributed runtimes where agents run on different machines. The ontology is purely logical -- it has no way to annotate where agents should be deployed or how they communicate at the transport level.

---

## 4. Proposed Extensions with YAML Examples

### 4.1 Channel Entity Type (addresses Gaps 1, 4)

Add a `channel` entity type representing a named, typed communication channel with pub/sub semantics.

```yaml
entity_types:
  channel:
    description: "A named communication channel with pub/sub semantics"
    required:
      id:           { type: string }
      label:        { type: string }
      channel_type: { type: "enum[topic, queue, broadcast, request_reply]" }
    optional:
      message_schema: { type: schema_ref, description: "Schema of messages on this channel" }
      retention:      { type: "enum[none, last, all, windowed]", default: none }
      reducer:        { type: string, description: "How concurrent writes merge (append, replace, custom)" }
      buffer_size:    { type: "integer | 'unbounded'", default: unbounded }
    visual:
      shape: parallelogram
      default_color: "#1abc9c"
      layer: entity
```

New edge types for pub/sub:

```yaml
edge_types:
  publish:
    description: "An entity publishes messages to a channel"
    required:
      from: { type: "agent | step", description: "The publisher" }
      to:   { type: channel, description: "The channel" }
    optional:
      label:  { type: string }
      filter: { type: string, description: "Condition for publishing" }

  subscribe:
    description: "An entity subscribes to messages from a channel"
    required:
      from: { type: channel, description: "The channel" }
      to:   { type: "agent | step", description: "The subscriber" }
    optional:
      label:   { type: string }
      filter:  { type: string, description: "Subscription filter expression" }
      activates: { type: boolean, default: true, description: "Whether message arrival triggers the subscriber" }
```

**Example usage (MetaGPT-style):**

```yaml
entities:
  - id: design_channel
    type: channel
    label: "Design Artifacts"
    channel_type: topic
    message_schema: DesignDocument
    retention: all

  - id: architect
    type: agent
    label: "Architect"
    model: gpt-4

  - id: engineer
    type: agent
    label: "Engineer"
    model: gpt-4

edges:
  - type: publish
    from: architect
    to: design_channel
    label: "Publish design docs"

  - type: subscribe
    from: design_channel
    to: engineer
    filter: "message.type == 'api_spec'"
    activates: true
```

### 4.2 Handoff Edge Type (addresses Gap 2)

```yaml
edge_types:
  handoff:
    description: "An agent transfers control and conversation context to another agent"
    required:
      from: { type: agent, description: "The handing-off agent" }
      to:   { type: agent, description: "The receiving agent" }
    optional:
      label:     { type: string }
      condition: { type: string, description: "When to hand off" }
      context:   { type: "enum[full, summary, none]", default: full, description: "How much conversation context transfers" }
      resumable: { type: boolean, default: false, description: "Whether control can return to the original agent" }
    visual:
      style: solid
      color: "#8e44ad"
      arrow: true
```

**Example usage (OpenAI Swarm-style):**

```yaml
entities:
  - id: triage_agent
    type: agent
    label: "Triage Agent"
    model: gpt-4
    system_prompt: "Route user requests to the appropriate specialist."

  - id: billing_agent
    type: agent
    label: "Billing Agent"
    model: gpt-4

  - id: tech_support_agent
    type: agent
    label: "Tech Support Agent"
    model: gpt-4

edges:
  - type: handoff
    from: triage_agent
    to: billing_agent
    condition: "user_intent == 'billing'"
    context: full

  - type: handoff
    from: triage_agent
    to: tech_support_agent
    condition: "user_intent == 'technical'"
    context: full
```

### 4.3 Team Entity Type (addresses Gap 5)

```yaml
entity_types:
  team:
    description: "A group of agents with a collective execution strategy"
    required:
      id:      { type: string }
      label:   { type: string }
      members: { type: "list<agent_ref>", description: "Agents in this team" }
      strategy:
        type: "enum[sequential, hierarchical, consensus, round_robin, dynamic]"
        description: "How tasks are distributed and executed"
    optional:
      manager:          { type: agent_ref, description: "Manager agent (required for hierarchical)" }
      delegation:       { type: boolean, default: false, description: "Whether members can delegate to each other" }
      speaker_selection:
        type: "enum[round_robin, llm_based, priority, random, custom]"
        description: "How the next speaker is chosen in conversational mode"
      max_rounds:       { type: integer }
      termination:      { type: string, description: "Composable termination condition" }
    visual:
      shape: rounded_rect_double
      default_color: "#2980b9"
      layer: entity
```

**Example usage (CrewAI-style):**

```yaml
entities:
  - id: research_team
    type: team
    label: "Research Team"
    members: [researcher, analyst, writer]
    strategy: hierarchical
    manager: project_lead
    delegation: true
    speaker_selection: llm_based
    termination: "all_tasks_complete OR max_rounds(10)"
```

### 4.4 Conversation Entity Type (addresses Gap 6)

```yaml
entity_types:
  conversation:
    description: "A multi-turn dialogue with structured history"
    required:
      id:    { type: string }
      label: { type: string }
    optional:
      participants:    { type: "list<node_ref>" }
      history_schema:  { type: schema_ref, description: "Schema for conversation turns" }
      max_turns:       { type: integer }
      persistence:     { type: "enum[ephemeral, session, persistent]", default: session }
      nesting:         { type: boolean, default: false, description: "Whether sub-conversations are allowed" }
    visual:
      shape: rounded_rect
      default_color: "#16a085"
      layer: entity
```

**Example usage (CAMEL-style):**

```yaml
entities:
  - id: role_play_conversation
    type: conversation
    label: "Role-Play Session"
    participants: [ai_user_agent, ai_assistant_agent]
    history_schema: ConversationTurn
    max_turns: 50
    persistence: session

schemas:
  - name: ConversationTurn
    fields:
      - { name: speaker, type: string }
      - { name: role, type: string }
      - { name: content, type: string }
      - { name: turn_number, type: integer }
      - { name: termination_token, type: boolean }
```

### 4.5 Spec-Level State Schema (addresses Gap 3)

Extend `spec_shape` to include a typed state definition:

```yaml
spec_shape:
  optional:
    state:
      description: "Typed runtime state schema for this spec"
      properties:
        schema:   { type: schema_ref, description: "Pydantic-like state model" }
        channels:
          type: "list<channel_def>"
          item_schema:
            name:    { type: string }
            type:    { type: string }
            reducer: { type: "enum[replace, append, merge, custom]", default: replace }
        initial:  { type: object, description: "Default initial state values" }
```

**Example usage (LangGraph-style):**

```yaml
name: "Research Agent"
version: "1.0"
state:
  schema: AgentState
  channels:
    - { name: messages, type: "list<Message>", reducer: append }
    - { name: current_plan, type: Plan, reducer: replace }
    - { name: documents, type: "list<Document>", reducer: append }
  initial:
    messages: []
    current_plan: null
    documents: []

schemas:
  - name: AgentState
    fields:
      - { name: messages, type: "list<Message>" }
      - { name: current_plan, type: Plan }
      - { name: documents, type: "list<Document>" }
```

### 4.6 Composable Termination Conditions (addresses Gap 9)

Replace the simple `condition: string` on loops and protocols with a structured termination type:

```yaml
# New type usable in protocols, loops, teams
termination_condition:
  description: "A composable termination expression"
  variants:
    simple:    { condition: string }
    max_turns: { count: integer }
    max_time:  { duration: duration }
    text_match: { pattern: string, in_field: string }
    composite:
      operator: { type: "enum[and, or, not]" }
      conditions: { type: "list<termination_condition>" }
```

**Example usage:**

```yaml
processes:
  - id: debate_protocol
    type: protocol
    label: "Structured Debate"
    participants:
      - { entity: pro_agent, role: proponent }
      - { entity: con_agent, role: opponent }
      - { entity: judge, role: mediator }
    termination:
      operator: or
      conditions:
        - { max_turns: { count: 10 } }
        - { text_match: { pattern: "CONSENSUS_REACHED", in_field: "last_message" } }
        - operator: and
          conditions:
            - { simple: { condition: "pro_score > 8" } }
            - { simple: { condition: "con_score > 8" } }
```

---

## 5. Priority Ranking

| Priority | Gap | Proposed Extension | Rationale |
|----------|-----|--------------------|-----------|
| **CRITICAL** | No message/channel abstraction | Channel entity + publish/subscribe edges | Affects 4/6 frameworks. Fundamental to expressing any pub/sub, reactive, or event-driven multi-agent pattern. Without this, every multi-agent spec must manually wire all communication through step nodes. |
| **CRITICAL** | No agent handoff | Handoff edge type | Affects 3/6 frameworks. The handoff pattern is the core abstraction in OpenAI's agent SDK and CrewAI delegation. Currently impossible to express without faking it through orchestrator steps. |
| **IMPORTANT** | No typed state / state schema | Spec-level state definition with channels | Affects 2/6 frameworks, but also affects ALL specs in practice. Every existing spec uses untyped `state.data`. Adding typed state enables validation, code generation, and catches bugs at spec-write time. |
| **IMPORTANT** | No event-driven activation | `activates` flag on subscribe edges | Affects 3/6 frameworks. Follows naturally from adding channels. Essential for expressing MetaGPT and AutoGen patterns. |
| **IMPORTANT** | No team/crew entity | Team entity type | Affects 3/6 frameworks. CrewAI, MetaGPT, and AutoGen all treat agent groups as first-class. Currently the ontology is agent-centric, not team-centric. |
| **IMPORTANT** | No conversation abstraction | Conversation entity type | Affects 3/6 frameworks. Conversational multi-agent systems need conversation as a first-class object, not just a list in state.data. |
| **NICE-TO-HAVE** | No checkpointing/time-travel | Checkpoint metadata on spec_shape | Primarily LangGraph. Important for production systems but not an expressiveness gap per se -- it is more of a runtime concern. |
| **NICE-TO-HAVE** | No speaker selection strategy | speaker_selection field on team/protocol | Primarily AutoGen. Can be partially expressed via protocol rules today, but a dedicated field would be clearer. |
| **NICE-TO-HAVE** | No composable termination | Structured termination_condition type | Primarily AutoGen. Current string conditions work for simple cases. Composition is a convenience, not a blocker. |
| **NICE-TO-HAVE** | No deployment topology | Out of scope for ontology | Purely a runtime concern. The ontology should remain logical/architectural. |

---

## 6. Cross-Cutting Observations

### 6.1 The Orchestrator Bottleneck

The ontology's current design assumes a **central orchestration spine** -- a chain of `step` -> `gate` -> `step` processes with agents invoked from the side. This works for pipeline architectures but becomes awkward for:

- **Peer-to-peer** patterns (CAMEL, Swarm handoffs) where agents talk directly
- **Pub/sub** patterns (MetaGPT, AutoGen) where agents react to messages
- **Conversational** patterns where the conversation *is* the control flow

The channel + handoff extensions address this by allowing decentralized communication alongside the existing orchestrated patterns.

### 6.2 Static vs. Dynamic Topology

The ontology defines a **static graph** at spec-write time. All edges are pre-wired. The `spawn` process allows dynamic agent creation, but not dynamic edge creation. In practice, several frameworks support:

- **Dynamic delegation**: Agent A decides at runtime to ask agent B (CrewAI)
- **Dynamic handoff**: Agent A routes to agent B or C based on conversation (OpenAI Agents SDK)
- **Dynamic team formation**: Agents join/leave teams at runtime (not yet in any major framework, but emerging)

The handoff edge type with `condition` partially addresses this, but a more general "dynamic edge" concept may be needed in future versions.

### 6.3 Conversation as Control Flow

In CAMEL and AutoGen, the **conversation itself IS the execution trace**. There is no separate "process flow" -- agents take turns in a conversation, and the conversation progresses the task. The ontology currently separates "processes" (control flow) from "entities" (agents, stores). For conversational architectures, this separation creates friction because the conversation is simultaneously the data AND the control flow.

The proposed `conversation` entity type plus `channel` abstraction together address this, but it represents a philosophical shift: the ontology would need to support both **orchestrated** and **conversational** execution models.

### 6.4 What the Ontology Already Does Well

It is worth noting that several patterns that might seem like gaps are actually already expressible:

- **Recursive composition**: `spawn(template: 'self')` handles this cleanly
- **Fan-out/fan-in**: `spawn` with `aggregation` covers MapReduce, parallel workers
- **Human-in-the-loop**: `checkpoint` process type is well-designed
- **Guardrails/policies**: `policy` with `targets` and `effect` is more expressive than most frameworks
- **N-ary protocols**: `protocol` with `participants` and `roles` can express debates, voting, negotiation
- **Blackboard pattern**: `store(store_type: blackboard)` exists, though it lacks reactive semantics

---

## 7. Recommended Implementation Order

1. **Phase 1 (v0.2)**: Add `channel` entity type + `publish`/`subscribe` edge types + `handoff` edge type. These address the two CRITICAL gaps and unlock the largest set of new patterns.

2. **Phase 2 (v0.3)**: Add `state` to `spec_shape` with typed channels and reducers. Add `team` entity type. These address the IMPORTANT gaps and improve spec authoring quality.

3. **Phase 3 (v0.4)**: Add `conversation` entity type. Add structured `termination_condition` type. Add `speaker_selection` to `protocol`/`team`. These address the remaining IMPORTANT and NICE-TO-HAVE gaps.

---

## Sources

- [CAMEL: Communicative Agents for "Mind" Exploration (arXiv)](https://arxiv.org/abs/2303.17760)
- [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework (arXiv)](https://arxiv.org/html/2308.00352v6)
- [MetaGPT Overview (IBM)](https://www.ibm.com/think/topics/metagpt)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph Multi-Agent Orchestration Guide](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/langgraph-multi-agent-orchestration-complete-framework-guide-architecture-analysis-2025)
- [OpenAI Agents SDK (GitHub)](https://github.com/openai/openai-agents-python)
- [OpenAI Orchestrating Agents: Routines and Handoffs](https://cookbook.openai.com/examples/orchestrating_agents)
- [CrewAI Documentation: Flows](https://docs.crewai.com/en/concepts/flows)
- [CrewAI Framework 2025 Review](https://latenode.com/blog/ai-frameworks-technical-infrastructure/crewai-framework/crewai-framework-2025-complete-review-of-the-open-source-multi-agent-ai-platform)
- [AutoGen v0.4: Reimagining Agentic AI (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)
- [AutoGen Conversation Patterns](https://microsoft.github.io/autogen/0.2/docs/tutorial/conversation-patterns/)
- [AutoGen Distributed Agent Runtime](https://microsoft.github.io/autogen/stable//user-guide/core-user-guide/framework/distributed-agent-runtime.html)
- [LangGraph vs CrewAI vs AutoGen Guide for 2026](https://dev.to/pockit_tools/langgraph-vs-crewai-vs-autogen-the-complete-multi-agent-ai-orchestration-guide-for-2026-2d63)
