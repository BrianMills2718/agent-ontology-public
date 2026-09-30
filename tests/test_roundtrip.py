#!/usr/bin/env python3
"""Round-trip tests for import → validate → instantiate pipeline.

Tests that imported specs from LangGraph and CrewAI source files can be:
1. Imported to a YAML spec
2. Validated (0 errors)
3. Instantiated to valid Python code (syntax check)

For full E2E (running the agent), use test_agents.py with --spec flag.
"""

import ast
import sys
import tempfile
from pathlib import Path

import yaml

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_ontology"))

from agent_ontology.import_langgraph import import_langgraph
from agent_ontology.import_autogen import import_autogen
from agent_ontology.import_openai_agents import import_openai_agents
from agent_ontology.import_google_adk import import_google_adk
from agent_ontology.validate import validate_spec, load_yaml, ONTOLOGY_PATH

_ontology = load_yaml(ONTOLOGY_PATH)


# ═══════════════════════════════════════════════════════════════
# Test LangGraph Files
# ═══════════════════════════════════════════════════════════════

LG_CLASS_BASED = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: list[str]
    result: str

class MyAgent:
    def __init__(self):
        self.graph = StateGraph(AgentState)
        self.graph.add_node("process", self.process_node)
        self.graph.add_node("analyze", self.analyze_node)
        self.graph.add_edge(START, "process")
        self.graph.add_edge("process", "analyze")
        self.graph.add_edge("analyze", END)

    def process_node(self, state):
        return {"messages": state["messages"] + ["processed"]}

    def analyze_node(self, state):
        return {"result": "analysis complete"}
"""

LG_ADD_SEQUENCE = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    data: str

def step_a(state): return {"data": "a"}
def step_b(state): return {"data": "b"}
def step_c(state): return {"data": "c"}

graph = StateGraph(State)
graph.add_sequence(["step_a", "step_b", "step_c"])
graph.add_edge(START, "step_a")
graph.add_edge("step_c", END)
"""

LG_SEND_FANOUT = """\
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class State(TypedDict):
    chunks: list[str]
    results: Annotated[list[str], operator.add]

def split_input(state):
    return {"chunks": ["chunk1", "chunk2"]}

def process_chunk(state):
    return {"results": [f"processed"]}

def aggregate(state):
    return {"results": state["results"]}

def route_chunks(state):
    return [Send("process_chunk", {"chunks": [c]}) for c in state["chunks"]]

graph = StateGraph(State)
graph.add_node("split_input", split_input)
graph.add_node("process_chunk", process_chunk)
graph.add_node("aggregate", aggregate)
graph.add_edge(START, "split_input")
graph.add_conditional_edges("split_input", route_chunks)
graph.add_edge("process_chunk", "aggregate")
graph.add_edge("aggregate", END)
"""

LG_INTERRUPT = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

class State(TypedDict):
    query: str
    result: str

def process_query(state):
    return {"result": "processed"}

def human_review(state):
    approval = interrupt({"question": "Approve?"})
    return {"result": state["result"]}

def finalize(state):
    return {"result": state["result"]}

graph = StateGraph(State)
graph.add_node("process_query", process_query)
graph.add_node("human_review", human_review)
graph.add_node("finalize", finalize)
graph.add_edge(START, "process_query")
graph.add_edge("process_query", "human_review")
graph.add_edge("human_review", "finalize")
graph.add_edge("finalize", END)
app = graph.compile(interrupt_before=["human_review"])
"""

LG_DATACLASS = """\
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END

@dataclass
class AgentState:
    query: str = ""
    result: str = ""
    iterations: int = 0

def search(state):
    return {"result": "found"}

graph = StateGraph(AgentState)
graph.add_node("search", search)
graph.add_edge(START, "search")
graph.add_edge("search", END)
"""

LG_SUBGRAPH = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class InnerState(TypedDict):
    data: str

class OuterState(TypedDict):
    input: str
    output: str

def inner_process(state):
    return {"data": "processed"}

inner_graph = StateGraph(InnerState)
inner_graph.add_node("inner_process", inner_process)
inner_graph.add_edge(START, "inner_process")
inner_graph.add_edge("inner_process", END)
inner_app = inner_graph.compile()

def prepare(state):
    return {"input": state["input"]}

def finalize(state):
    return {"output": "done"}

outer_graph = StateGraph(OuterState)
outer_graph.add_node("prepare", prepare)
outer_graph.add_node("sub_agent", inner_app)
outer_graph.add_node("finalize", finalize)
outer_graph.add_edge(START, "prepare")
outer_graph.add_edge("prepare", "sub_agent")
outer_graph.add_edge("sub_agent", "finalize")
outer_graph.add_edge("finalize", END)
"""

LG_CONDITIONAL = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    answer: str
    needs_search: bool

def reason(state):
    return {"needs_search": True}

def search(state):
    return {"answer": "found"}

def answer(state):
    return {"answer": "direct answer"}

def should_search(state):
    if state.get("needs_search"):
        return "search"
    return "answer"

graph = StateGraph(State)
graph.add_node("reason", reason)
graph.add_node("search", search)
graph.add_node("answer", answer)
graph.add_edge(START, "reason")
graph.add_conditional_edges("reason", should_search, {"search": "search", "answer": "answer"})
graph.add_edge("search", "answer")
graph.add_edge("answer", END)
"""

LG_COMMAND = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class State(TypedDict):
    data: str
    step: str

def router(state):
    if state.get("data"):
        return Command(goto="process")
    return Command(goto="collect")

def collect(state):
    return {"data": "collected"}

def process(state):
    return {"step": "processed"}

graph = StateGraph(State)
graph.add_node("router", router)
graph.add_node("collect", collect)
graph.add_node("process", process)
graph.add_edge(START, "router")
graph.add_edge("collect", "router")
graph.add_edge("process", END)
"""


def _import_and_check(source_code, test_name, expected_checks=None):
    """Import source code, validate spec, and check properties."""
    # Write source to temp file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(source_code)
        f.flush()
        source_path = f.name

    try:
        # Step 1: Import
        spec = import_langgraph(source_path)
        assert spec is not None, f"{test_name}: import returned None"
        assert "entry_point" in spec, f"{test_name}: missing entry_point"
        assert spec["entry_point"] is not None, f"{test_name}: entry_point is None"
        assert len(spec.get("processes", [])) > 0, f"{test_name}: no processes"

        # Step 2: Validate
        errors, warnings = validate_spec(spec, _ontology, source_path)
        assert len(errors) == 0, f"{test_name}: validation errors: {errors}"

        # Step 3: Check specific properties
        if expected_checks:
            for check_name, check_fn in expected_checks.items():
                assert check_fn(spec), f"{test_name}: check '{check_name}' failed"

        return spec, errors, warnings
    finally:
        Path(source_path).unlink(missing_ok=True)


def test_lg_class_based():
    """self.graph = StateGraph(...) is detected."""
    spec, _, _ = _import_and_check(LG_CLASS_BASED, "class_based", {
        "has_process": lambda s: any(p["id"] == "process" for p in s["processes"]),
        "has_analyze": lambda s: any(p["id"] == "analyze" for p in s["processes"]),
        "entry_point": lambda s: s["entry_point"] == "process",
        "has_schema": lambda s: any(sc["name"] == "AgentState" for sc in s.get("schemas", [])),
    })
    print(f"  PASS class_based: {len(spec['processes'])} processes, {len(spec['edges'])} edges")


def test_lg_add_sequence():
    """add_sequence() creates nodes and edges."""
    spec, _, _ = _import_and_check(LG_ADD_SEQUENCE, "add_sequence", {
        "3_processes": lambda s: len([p for p in s["processes"] if p["type"] == "step"]) == 3,
        "2_flow_edges": lambda s: len([e for e in s["edges"] if e["type"] == "flow"]) == 2,
        "entry_point": lambda s: s["entry_point"] == "step_a",
    })
    print(f"  PASS add_sequence: {len(spec['processes'])} processes, {len(spec['edges'])} edges")


def test_lg_send_fanout():
    """Send() creates fan-out edges."""
    spec, _, _ = _import_and_check(LG_SEND_FANOUT, "send_fanout", {
        "has_split": lambda s: any(p["id"] == "split_input" for p in s["processes"]),
        "has_process_chunk": lambda s: any(p["id"] == "process_chunk" for p in s["processes"]),
        "fan_out_edge": lambda s: any(
            e["from"] == "split_input" and e["to"] == "process_chunk" for e in s["edges"]
        ),
    })
    print(f"  PASS send_fanout: {len(spec['processes'])} processes, {len(spec['edges'])} edges")


def test_lg_interrupt():
    """interrupt() creates checkpoint process."""
    spec, _, _ = _import_and_check(LG_INTERRUPT, "interrupt", {
        "has_checkpoint": lambda s: any(p["type"] == "checkpoint" for p in s["processes"]),
        "has_human": lambda s: any(e["type"] == "human" for e in s.get("entities", [])),
    })
    print(f"  PASS interrupt: {len(spec['processes'])} processes, {len(spec['edges'])} edges")


def test_lg_dataclass():
    """@dataclass state is extracted."""
    spec, _, _ = _import_and_check(LG_DATACLASS, "dataclass", {
        "has_schema": lambda s: any(sc["name"] == "AgentState" for sc in s.get("schemas", [])),
        "schema_has_fields": lambda s: len(next(
            sc["fields"] for sc in s.get("schemas", []) if sc["name"] == "AgentState"
        )) == 3,
    })
    print(f"  PASS dataclass: {len(spec.get('schemas', []))} schemas")


def test_lg_subgraph():
    """Subgraph-as-node is detected with description."""
    spec, _, _ = _import_and_check(LG_SUBGRAPH, "subgraph", {
        "has_sub_agent": lambda s: any(p["id"] == "sub_agent" for p in s["processes"]),
        "sub_agent_description": lambda s: any(
            p["id"] == "sub_agent" and "subgraph" in p.get("description", "").lower()
            for p in s["processes"]
        ),
    })
    print(f"  PASS subgraph: {len(spec['processes'])} processes")


def test_lg_conditional():
    """Conditional edges create gate processes."""
    spec, _, _ = _import_and_check(LG_CONDITIONAL, "conditional", {
        "has_gate": lambda s: any(p["type"] == "gate" for p in s["processes"]),
        "gate_has_branches": lambda s: any(
            p["type"] == "gate" and len(p.get("branches", [])) >= 2
            for p in s["processes"]
        ),
    })
    print(f"  PASS conditional: {len(spec['processes'])} processes")


def test_lg_command():
    """Command(goto=...) creates edges."""
    spec, _, _ = _import_and_check(LG_COMMAND, "command", {
        "router_to_process": lambda s: any(
            e["from"] == "router" and e["to"] == "process" for e in s["edges"]
        ),
        "router_to_collect": lambda s: any(
            e["from"] == "router" and e["to"] == "collect" for e in s["edges"]
        ),
    })
    print(f"  PASS command: {len(spec['processes'])} processes, {len(spec['edges'])} edges")


# ═══════════════════════════════════════════════════════════════
# Test AutoGen Files
# ═══════════════════════════════════════════════════════════════

AG_V02_GROUPCHAT = """\
import autogen

llm_config = {"model": "gpt-4o", "temperature": 0}

assistant = autogen.AssistantAgent(
    name="research_assistant",
    system_message="You are a helpful research assistant.",
    llm_config=llm_config,
)

coder = autogen.AssistantAgent(
    name="coder",
    system_message="You write Python code to solve problems.",
    llm_config=llm_config,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    code_execution_config={"work_dir": "coding", "use_docker": False},
)

@user_proxy.register_for_execution()
@assistant.register_for_llm(description="Search the web")
def web_search(query: str) -> str:
    return f"Results for: {query}"

groupchat = autogen.GroupChat(
    agents=[assistant, coder, user_proxy],
    messages=[],
    max_round=12,
    speaker_selection_method="auto",
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
)

user_proxy.initiate_chat(manager, message="Research sorting algorithms")
"""

AG_V04_TEAMS = """\
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.tools import FunctionTool
from autogen_ext.models import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")

def calculator(expression: str) -> str:
    return str(eval(expression))

calc_tool = FunctionTool(calculator, description="A calculator tool")

planner = AssistantAgent(
    name="planner",
    model_client=model_client,
    system_message="You plan tasks.",
    tools=[calc_tool],
)

executor = AssistantAgent(
    name="executor",
    model_client=model_client,
    system_message="You execute tasks.",
    handoffs=["planner"],
)

team = RoundRobinGroupChat(
    participants=[planner, executor],
    max_turns=10,
)
"""


def _import_and_check_autogen(source_code, test_name, expected_checks=None):
    """Import AutoGen source code, validate spec, and check properties."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(source_code)
        f.flush()
        source_path = f.name

    try:
        spec = import_autogen(Path(source_path))
        assert spec is not None, f"{test_name}: import returned None"
        assert "entry_point" in spec, f"{test_name}: missing entry_point"
        assert len(spec.get("processes", [])) > 0, f"{test_name}: no processes"

        errors, warnings = validate_spec(spec, _ontology, source_path)
        assert len(errors) == 0, f"{test_name}: validation errors: {errors}"

        if expected_checks:
            for check_name, check_fn in expected_checks.items():
                assert check_fn(spec), f"{test_name}: check '{check_name}' failed"

        return spec, errors, warnings
    finally:
        Path(source_path).unlink(missing_ok=True)


def test_ag_v02_groupchat():
    """AutoGen v0.2 GroupChat with tool registration."""
    spec, _, _ = _import_and_check_autogen(AG_V02_GROUPCHAT, "ag_v02_groupchat", {
        "has_agents": lambda s: len([e for e in s["entities"] if e["type"] == "agent"]) >= 3,
        "has_tool": lambda s: any(e["type"] == "tool" for e in s["entities"]),
        "has_team": lambda s: any(e["type"] == "team" for e in s["entities"]),
        "has_handoff_or_invoke": lambda s: any(
            e["type"] in ("handoff", "invoke") for e in s["edges"]
        ),
    })
    print(f"  PASS ag_v02_groupchat: {len(spec['entities'])} entities, {len(spec['edges'])} edges")


def test_ag_v04_teams():
    """AutoGen v0.4 with FunctionTool and teams."""
    spec, _, _ = _import_and_check_autogen(AG_V04_TEAMS, "ag_v04_teams", {
        "has_agents": lambda s: len([e for e in s["entities"] if e["type"] == "agent"]) >= 2,
        "has_tool": lambda s: any(e["type"] == "tool" for e in s["entities"]),
        "has_team": lambda s: any(e["type"] == "team" for e in s["entities"]),
    })
    print(f"  PASS ag_v04_teams: {len(spec['entities'])} entities, {len(spec['edges'])} edges")


# ═══════════════════════════════════════════════════════════════
# Test OpenAI Agents SDK Files
# ═══════════════════════════════════════════════════════════════

OAI_HANDOFF = """\
from agents import Agent, Runner, function_tool, WebSearchTool

@function_tool
def get_weather(city: str) -> str:
    \"\"\"Get the weather for a city.\"\"\"
    return f"Weather in {city}: sunny"

web_search = WebSearchTool()

billing_agent = Agent(
    name="billing_agent",
    instructions="You handle billing.",
    model="gpt-4o",
    tools=[get_weather],
)

support_agent = Agent(
    name="support_agent",
    instructions="You handle support.",
    model="gpt-4o",
    tools=[web_search],
)

triage_agent = Agent(
    name="triage_agent",
    instructions="Route to specialists.",
    model="gpt-4o",
    handoffs=[billing_agent, support_agent],
)

billing_agent.handoffs.append(triage_agent)

result = Runner.run_sync(triage_agent, "billing question")
"""

OAI_PIPELINE = """\
from pydantic import BaseModel
from agents import Agent, Runner, function_tool, input_guardrail, GuardrailFunctionOutput
import asyncio

class AnalysisResult(BaseModel):
    findings: str
    confidence: float

@function_tool
def search_db(query: str) -> str:
    \"\"\"Search the database.\"\"\"
    return "results"

@input_guardrail
async def safety_check(ctx, agent, input):
    return GuardrailFunctionOutput(output_info="safe", tripwire_triggered=False)

researcher = Agent(
    name="researcher",
    instructions="Research topics.",
    model="gpt-4o",
    tools=[search_db],
    output_type=AnalysisResult,
    input_guardrails=[safety_check],
)

writer = Agent(name="writer", instructions="Write content.", model="gpt-4o")
critic_a = Agent(name="critic_a", instructions="Critique technically.", model="gpt-4o")
critic_b = Agent(name="critic_b", instructions="Critique readability.", model="gpt-4o")

async def main():
    r1 = await Runner.run(researcher, "quantum computing")
    r2 = await Runner.run(writer, r1.final_output)
    ca, cb = await asyncio.gather(
        Runner.run(critic_a, r2.final_output),
        Runner.run(critic_b, r2.final_output),
    )

asyncio.run(main())
"""

OAI_ORCHESTRATOR = """\
from agents import Agent, Runner

sub_agent_a = Agent(name="sub_a", instructions="Do task A.", model="gpt-4o")
sub_agent_b = Agent(name="sub_b", instructions="Do task B.", model="gpt-4o")

orchestrator = Agent(
    name="orchestrator",
    instructions="Coordinate sub-agents.",
    model="gpt-4o",
    tools=[
        sub_agent_a.as_tool(tool_name="task_a", tool_description="Delegate task A"),
        sub_agent_b.as_tool(tool_name="task_b", tool_description="Delegate task B"),
    ],
)

result = Runner.run_sync(orchestrator, "do both tasks")
"""


def _import_and_check_oai(source_code, test_name, expected_checks=None):
    """Import OpenAI Agents SDK source code, validate spec, and check properties."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(source_code)
        f.flush()
        source_path = f.name

    try:
        spec = import_openai_agents(Path(source_path))
        assert spec is not None, f"{test_name}: import returned None"
        assert "entry_point" in spec, f"{test_name}: missing entry_point"
        assert len(spec.get("processes", [])) > 0, f"{test_name}: no processes"

        errors, warnings = validate_spec(spec, _ontology, source_path)
        assert len(errors) == 0, f"{test_name}: validation errors: {errors}"

        if expected_checks:
            for check_name, check_fn in expected_checks.items():
                assert check_fn(spec), f"{test_name}: check '{check_name}' failed"

        return spec, errors, warnings
    finally:
        Path(source_path).unlink(missing_ok=True)


def test_oai_handoff():
    """OpenAI Agents SDK handoff pattern."""
    spec, _, _ = _import_and_check_oai(OAI_HANDOFF, "oai_handoff", {
        "has_3_agents": lambda s: len([e for e in s["entities"] if e["type"] == "agent"]) == 3,
        "has_tools": lambda s: len([e for e in s["entities"] if e["type"] == "tool"]) == 2,
        "has_handoff_edges": lambda s: len([e for e in s["edges"] if e["type"] == "handoff"]) >= 2,
        "has_invoke": lambda s: any(e["type"] == "invoke" for e in s["edges"]),
    })
    print(f"  PASS oai_handoff: {len(spec['entities'])} entities, {len(spec['edges'])} edges")


def test_oai_pipeline():
    """OpenAI Agents SDK pipeline with guardrails and fan-out."""
    spec, _, _ = _import_and_check_oai(OAI_PIPELINE, "oai_pipeline", {
        "has_4_agents": lambda s: len([e for e in s["entities"] if e["type"] == "agent"]) == 4,
        "has_tool": lambda s: any(e["type"] == "tool" for e in s["entities"]),
        "has_policy": lambda s: any(p["type"] == "policy" for p in s["processes"]),
        "has_fan_out": lambda s: any(p["id"].startswith("fan_out") for p in s["processes"]),
        "has_schema": lambda s: any(sc["name"] == "AnalysisResult" for sc in s.get("schemas", [])),
    })
    print(f"  PASS oai_pipeline: {len(spec['entities'])} entities, {len(spec['processes'])} processes")


def test_oai_orchestrator():
    """OpenAI Agents SDK agent-as-tool orchestrator pattern."""
    spec, _, _ = _import_and_check_oai(OAI_ORCHESTRATOR, "oai_orchestrator", {
        "has_3_agents": lambda s: len([e for e in s["entities"] if e["type"] == "agent"]) == 3,
        "has_invoke": lambda s: any(e["type"] == "invoke" for e in s["edges"]),
    })
    print(f"  PASS oai_orchestrator: {len(spec['entities'])} entities, {len(spec['edges'])} edges")


# ═══════════════════════════════════════════════════════════════
# Test Google ADK Files
# ═══════════════════════════════════════════════════════════════

ADK_SIMPLE = """\
from google.adk.agents import Agent
from google.adk.tools import google_search

def get_weather(city: str) -> dict:
    \"\"\"Get current weather for a city.\"\"\"
    return {"temp": 72, "condition": "sunny"}

weather_agent = Agent(
    name="weather_agent",
    model="gemini-2.5-flash",
    instruction="You are a weather assistant. Use tools to find weather info.",
    tools=[get_weather, google_search],
    description="Provides weather information",
)
"""

ADK_PIPELINE = """\
from pydantic import BaseModel
from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.tools import exit_loop, google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

class AnalysisResult(BaseModel):
    findings: str
    confidence: float

def search_papers(query: str) -> dict:
    \"\"\"Search academic papers.\"\"\"
    return {"papers": ["paper1", "paper2"]}

researcher = Agent(
    name="researcher",
    model="gemini-2.5-flash",
    instruction="Research the topic: {topic}",
    tools=[search_papers, google_search],
    output_key="research_findings",
    output_schema=AnalysisResult,
)

critic = Agent(
    name="critic",
    model="gemini-2.5-flash",
    instruction="Critique the research: {research_findings}",
    output_key="criticism",
)

refiner = Agent(
    name="refiner",
    model="gemini-2.5-flash",
    instruction="Refine based on criticism: {criticism}. If quality is sufficient, exit loop.",
    tools=[exit_loop],
    output_key="refined_output",
)

writer_a = Agent(
    name="writer_a",
    model="gemini-2.5-flash",
    instruction="Write a technical summary from: {research_findings}",
    output_key="summary_a",
)

writer_b = Agent(
    name="writer_b",
    model="gemini-2.5-flash",
    instruction="Write a plain-language summary from: {research_findings}",
    output_key="summary_b",
)

synthesizer = Agent(
    name="synthesizer",
    model="gemini-2.5-flash",
    instruction="Combine summaries: {summary_a} and {summary_b}",
    output_key="final_output",
)

refine_loop = LoopAgent(
    name="refine_loop",
    max_iterations=3,
    sub_agents=[critic, refiner],
)

parallel_writing = ParallelAgent(
    name="parallel_writing",
    sub_agents=[writer_a, writer_b],
)

root_agent = SequentialAgent(
    name="research_pipeline",
    sub_agents=[researcher, refine_loop, parallel_writing, synthesizer],
    description="Research pipeline with refinement and parallel writing",
)

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name="research", session_service=session_service)
"""


def _import_and_check_adk(source_code, test_name, expected_checks=None):
    """Import Google ADK source code, validate spec, and check properties."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(source_code)
        f.flush()
        source_path = f.name

    try:
        spec = import_google_adk(Path(source_path))
        assert spec is not None, f"{test_name}: import returned None"
        assert "entry_point" in spec, f"{test_name}: missing entry_point"
        assert len(spec.get("processes", [])) > 0, f"{test_name}: no processes"

        errors, warnings = validate_spec(spec, _ontology, source_path)
        assert len(errors) == 0, f"{test_name}: validation errors: {errors}"

        if expected_checks:
            for check_name, check_fn in expected_checks.items():
                assert check_fn(spec), f"{test_name}: check '{check_name}' failed"

        return spec, errors, warnings
    finally:
        Path(source_path).unlink(missing_ok=True)


def test_adk_simple():
    """Google ADK simple agent with tools."""
    spec, _, _ = _import_and_check_adk(ADK_SIMPLE, "adk_simple", {
        "has_agent": lambda s: any(e["type"] == "agent" for e in s["entities"]),
        "has_tools": lambda s: len([e for e in s["entities"] if e["type"] == "tool"]) == 2,
        "has_step": lambda s: any(p["type"] == "step" for p in s["processes"]),
        "has_invoke": lambda s: any(e["type"] == "invoke" for e in s["edges"]),
    })
    print(f"  PASS adk_simple: {len(spec['entities'])} entities, {len(spec['edges'])} edges")


def test_adk_pipeline():
    """Google ADK pipeline with sequential, parallel, and loop agents."""
    spec, _, _ = _import_and_check_adk(ADK_PIPELINE, "adk_pipeline", {
        "has_agents": lambda s: len([e for e in s["entities"] if e["type"] == "agent"]) >= 6,
        "has_gate": lambda s: any(p["type"] == "gate" for p in s["processes"]),
        "has_loop_edge": lambda s: any(e["type"] == "loop" for e in s["edges"]),
        "has_fanout": lambda s: any("parallel" in p["id"].lower() for p in s["processes"]),
        "has_schema": lambda s: any(sc["name"] == "AnalysisResult" for sc in s.get("schemas", [])),
        "entry_is_researcher": lambda s: s["entry_point"] == "run_researcher",
    })
    print(f"  PASS adk_pipeline: {len(spec['entities'])} entities, {len(spec['processes'])} processes, {len(spec['edges'])} edges")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

ALL_TESTS = [
    # LangGraph
    test_lg_class_based,
    test_lg_add_sequence,
    test_lg_send_fanout,
    test_lg_interrupt,
    test_lg_dataclass,
    test_lg_subgraph,
    test_lg_conditional,
    test_lg_command,
    # AutoGen
    test_ag_v02_groupchat,
    test_ag_v04_teams,
    # OpenAI Agents SDK
    test_oai_handoff,
    test_oai_pipeline,
    test_oai_orchestrator,
    # Google ADK
    test_adk_simple,
    test_adk_pipeline,
]


def main():
    print("Round-trip Import Tests")
    print("=" * 50)
    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
