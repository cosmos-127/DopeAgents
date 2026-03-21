# Architecture

## Overview

DopeAgents is a Python library that ships production-grade AI agents as internally sophisticated multi-step workflows with externally simple typed interfaces. This document describes the system architecture — how components are organized, how they interact, and why the boundaries exist where they do.

The central architectural insight is **separation of workflow logic from infrastructure concerns**. Agents contain only their domain-specific multi-step workflow. Everything else — cost tracking, budget enforcement, retry, observability, MCP exposure — lives in surrounding layers that agents never interact with directly.

---

## System Layers

The system is organized into five distinct layers, each with a clear responsibility boundary:

```
┌─────────────────────────────────────────────────────────────────┐
│                      External Interface Layer                    │
│                                                                  │
│   Plain Python (.run())          MCP Protocol (.as_mcp_tool())   │
│   CLI (dopeagents run)           MCP Server (dopeagents mcp)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                      Lifecycle Layer                             │
│                                                                  │
│   AgentExecutor    CostTracker    BudgetConfig    RetryPolicy    │
│   ConsoleTracer    OTelTracer     DegradationChain               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                      Agent Layer                                 │
│                                                                  │
│   Agent[InputT, OutputT] base class                              │
│   DeepSummarizer    ResearchAgent    (user-defined agents)       │
│   Pydantic I/O contracts    step_prompts    _build_graph()       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                      Extraction Layer                            │
│                                                                  │
│   self._extract()  →  Instructor  →  LiteLLM  →  LLM Provider   │
│   Schema validation    Provider routing    Cost metadata          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────────────┐
│                      Orchestration Layer (Internal)              │
│                                                                  │
│   LangGraph StateGraph    Conditional edges    Refinement loops   │
│   (Private implementation detail — never exposed to users)       │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: Orchestration Layer (Internal)

**Responsibility:** Stateful graph execution with conditional branching and loops.

**Implementation:** LangGraph `StateGraph`. Each agent defines its internal workflow as a directed graph of step methods connected by edges and conditional edges.

**Key design decision:** This layer is a **private implementation detail**. No public API exposes graph nodes, edges, or state schemas. Users never import LangGraph, never interact with the graph directly, and never need to understand graph concepts. If a better orchestration primitive emerges, agents can migrate internally without breaking any external interface.

**What lives here:**

- `StateGraph` construction in each agent's `_build_graph()` method
- Graph state schemas (internal `TypedDict` definitions)
- Conditional edge functions (e.g., `_should_refine()`)
- Edge wiring between step methods

**What does NOT live here:**

- Step logic (lives in the Agent Layer)
- Cost tracking, retry, observability (lives in the Lifecycle Layer)
- Input/output contracts (lives in the Agent Layer)

### Layer 2: Extraction Layer

**Responsibility:** Structured LLM interaction — sending prompts, receiving validated typed responses, routing across providers, and surfacing cost/token metadata.

**Implementation:** Instructor for structured extraction with automatic schema validation and retry. LiteLLM for provider routing across 100+ LLM providers with pricing-aware cost metadata.

**Key design decision:** Every LLM interaction in every agent goes through a single primitive: `self._extract()`. This method is defined on the `Agent` base class and wraps `instructor.from_litellm(completion)`. No agent step ever calls an LLM directly, parses JSON manually, or handles malformed output. This guarantees that every LLM call in every step gets:

- Pydantic schema validation with automatic re-ask on validation failure
- Provider-agnostic routing (same agent works on OpenAI, Anthropic, Google, Ollama, etc.)
- Cost and token metadata attached to every response
- Consistent extraction mode selection per provider

```
Agent step method
       │
       ▼
self._extract(response_model=T, messages=[...])
       │
       ▼
instructor.from_litellm(litellm.completion)
       │
       ├── Schema validation (Instructor)
       │     └── Re-ask on validation failure (automatic)
       │
       ├── Provider routing (LiteLLM)
       │     └── Model string → correct API endpoint
       │
       └── Cost metadata (LiteLLM)
             └── response_cost, token counts attached to response
```

**What lives here:**

- `_extract()` and `_extract_partial()` base class methods
- Instructor client initialization and mode selection
- LiteLLM model string parsing and provider routing
- Per-call cost and token metadata surfacing

**What does NOT live here:**

- Prompt content (lives in the Agent Layer as `step_prompts`)
- Cost aggregation and budget enforcement (lives in the Lifecycle Layer)
- Retry for infrastructure failures like timeouts (lives in the Lifecycle Layer)

### Layer 3: Agent Layer

**Responsibility:** Domain-specific multi-step workflow logic. Each agent defines what steps to execute, in what order, with what prompts, producing what typed output.

**Implementation:** The `Agent[InputT, OutputT]` generic base class. Concrete agents subclass this, declare their metadata and step prompts, define `_build_graph()`, implement step methods, and implement `run()`.

**Key design decision:** Agents contain **only workflow logic**. No infrastructure code appears in any agent — no cost tracking, no retry logic, no observability spans, no JSON parsing, no provider-specific code. Each step method contains exactly one concern: its domain logic, expressed as a call to `_extract()` with a prompt and a response schema.

```python
# This is the ENTIRE implementation of a step. Nothing else.
def _evaluate_quality(self, state: dict) -> dict:
    evaluation = self._extract(
        response_model=QualityEvaluation,
        messages=[
            {"role": "system", "content": self.step_prompts["evaluate"]},
            {"role": "user", "content": f"Original: {state['document'][:3000]}\n\nSummary: {state['draft_summary']}"},
        ],
    )
    return {"quality_score": evaluation.score, "feedback": evaluation.feedback}
```

**What lives here:**

- `Agent[InputT, OutputT]` base class with generic type parameters
- Pydantic input/output models (external contracts)
- Pydantic step schemas (internal, per-step response models)
- `step_prompts` class variable — all prompts declared as inspectable data
- `_build_graph()` — graph construction (delegates to Orchestration Layer)
- Step methods — domain logic calling `_extract()`
- `run()` — public entry point, invokes graph, maps final state to output model
- `debug()` — returns graph structure, step prompts, and step schemas without execution
- `describe()` — returns agent metadata for discovery and composition
- Class metadata: `name`, `version`, `description`, `capabilities`, `tags`

**What does NOT live here:**

- Any infrastructure concern (cost, retry, tracing, caching, budget)
- Any framework-specific code beyond the `_extract()` call
- Any direct LLM interaction (always goes through `_extract()`)

### Layer 4: Lifecycle Layer

**Responsibility:** Production infrastructure that wraps agent execution — cost tracking, budget enforcement, retry, observability, caching, PII redaction, and graceful degradation.

**Implementation:** The `AgentExecutor` class, along with supporting components: `CostTracker`, `BudgetConfig`, `RetryPolicy`, `DegradationChain`, tracer implementations, and cache implementations.

**Key design decision:** The Lifecycle Layer wraps agents **from the outside**. Agents are unaware of its existence. This means:

- Agents can be tested without any lifecycle infrastructure (just call `run()` directly)
- Lifecycle policies can be changed per-execution without modifying the agent
- Multiple lifecycle configurations can wrap the same agent instance
- New lifecycle concerns (e.g., rate limiting, A/B testing) can be added without touching any agent code

```
AgentExecutor.run(agent, input, context)
       │
       ├── Input validation (Pydantic)
       ├── Budget check (pre-execution)
       ├── Start trace span
       │
       ▼
   agent.run(input, context)
       │
       ├── [per step] ──► Cost accumulation (from _extract() metadata)
       ├── [per step] ──► Step-level trace span
       ├── [per step] ──► Budget guard check
       ├── [per step] ──► Retry on transient failure
       │
       ▼
   AgentExecutionResult
       │
       ├── output: OutputT (typed agent output)
       ├── metrics: ExecutionMetrics
       │     ├── total_cost_usd
       │     ├── total_latency_ms
       │     └── steps: list[StepMetrics]
       │           ├── name, cost, latency_ms, tokens
       │           └── metadata (quality scores, etc.)
       ├── Close trace span
       └── Accumulate into CostTracker
```

**Step-level interception** is the critical capability. The Lifecycle Layer doesn't just wrap the entire `run()` call — it hooks into each internal graph step. This is achieved by instrumenting the `_extract()` base class method. Every `_extract()` call reports its cost, tokens, latency, and step name to the Lifecycle Layer. This enables:

- **Per-step cost tracking:** "The evaluate step cost $0.0089, the refine step cost $0.0178"
- **Per-step budget guards:** "No single step can exceed $0.05"
- **Per-step retry:** "The synthesize step hit a rate limit — retry it, but don't re-run steps 1-3"
- **Per-step observability:** "OpenTelemetry child span for each step with cost and latency"

**Budget enforcement operates at three granularities:**

| Granularity   | Configuration         | Behavior                                                          |
| ------------- | --------------------- | ----------------------------------------------------------------- |
| Per-step      | `max_cost_per_step` | Prevents any single internal step from exceeding a cost threshold |
| Per-agent-run | `max_cost_per_call` | Caps total cost across all steps in one `run()` invocation      |
| Global        | `max_cost_global`   | Caps total cost across all agent runs in the session              |

When `on_exceeded="degrade"`, budget exhaustion during a refinement loop returns the best result produced so far rather than discarding all completed work.

**What lives here:**

- `AgentExecutor` — wraps `run()` with full lifecycle management
- `CostTracker` — accumulates cost data per step, per agent, and globally
- `BudgetConfig` — declares cost limits at multiple granularities
- `RetryPolicy` — step-level retry with exponential backoff for transient errors
- `DegradationChain` — agent-level fallback when entire agents fail
- Tracer implementations (`ConsoleTracer`, `OTelTracer`) — automatic span management
- PII redaction — fields marked as PII in Pydantic schemas are scrubbed from observability output
- Cache layer — deterministic steps and repeated inputs can be cached

**What does NOT live here:**

- Any workflow logic (lives in the Agent Layer)
- Any prompt content (lives in the Agent Layer)
- Any LLM interaction (lives in the Extraction Layer)

### Layer 5: External Interface Layer

**Responsibility:** Exposing agents to the outside world through multiple interaction modes.

**Implementation:** Four interfaces, all backed by the same agent instances:

```
┌─────────────────────────────────────────────────┐
│              External Interface Layer             │
│                                                   │
│  ┌───────────┐  ┌────────┐  ┌─────┐  ┌───────┐  │
│  │ .run()    │  │  MCP   │  │ CLI │  │ Debug │  │
│  │ (Python)  │  │ Server │  │     │  │ Mode  │  │
│  └─────┬─────┘  └───┬────┘  └──┬──┘  └───┬───┘  │
│        │            │          │          │       │
│        └────────────┴──────────┴──────────┘       │
│                         │                         │
│              Agent Layer + Lifecycle Layer         │
└─────────────────────────────────────────────────┘
```

**Plain Python (`.run()`):** Direct method call. The simplest interface — import the agent, construct input, call `run()`, get typed output. Optionally wrap with `AgentExecutor` for lifecycle management.

**MCP Server (`.as_mcp_tool()`):** Protocol-level exposure via FastMCP. Each agent registers as an MCP tool with auto-generated JSON Schema from its Pydantic input/output types. A single MCP tool call triggers the entire internal multi-step workflow. MCP clients see a simple tool with parameters and a response — they have no visibility into internal graph complexity.

**CLI (`dopeagents run`):** Thin shell over `.run()` with JSON input/output. Useful for scripting, CI/CD, and quick testing. `dopeagents dry-run` invokes debug mode without API calls.

**Debug Mode (`.debug()`):** Returns the complete internal structure — graph topology, step prompts, step schemas, message templates — without executing anything. No API calls, no cost. Used for inspection, testing prompt changes, and MCP prompt exposure.

**What lives here:**

- `.as_mcp_tool()` and `.as_mcp_server()` methods on the Agent base class
- `create_mcp_server()` factory for multi-agent MCP servers
- CLI entry points (`dopeagents list`, `run`, `dry-run`, `describe`, `mcp serve`)
- `.debug()` method returning `DebugInfo` with graph, prompts, and schemas
- MCP resource registration (`dopeagents://catalog`)

---

## Component Interactions

### Execution Flow: Direct Python

```
 User code                Agent                    Extraction              Orchestration
    │                       │                         │                       │
    │  run(input)           │                         │                       │
    │──────────────────────►│                         │                       │
    │                       │  _build_graph()         │                       │
    │                       │────────────────────────────────────────────────►│
    │                       │                         │                       │
    │                       │  graph.invoke(state)    │                       │
    │                       │────────────────────────────────────────────────►│
    │                       │                         │                       │
    │                       │  [step: analyze]        │                       │
    │                       │◄────────────────────────────────────────────────│
    │                       │                         │                       │
    │                       │  _extract(DocumentAnalysis, messages)           │
    │                       │────────────────────────►│                       │
    │                       │                         │  instructor → litellm │
    │                       │                         │  → LLM provider       │
    │                       │  DocumentAnalysis       │                       │
    │                       │◄────────────────────────│                       │
    │                       │                         │                       │
    │                       │  return step state      │                       │
    │                       │────────────────────────────────────────────────►│
    │                       │                         │                       │
    │                       │  [step: chunk]          │                       │
    │                       │◄────────────────────────────────────────────────│
    │                       │  (deterministic, no _extract)                   │
    │                       │────────────────────────────────────────────────►│
    │                       │                         │                       │
    │                       │  ... remaining steps ...│                       │
    │                       │                         │                       │
    │                       │  [step: evaluate]       │                       │
    │                       │◄────────────────────────────────────────────────│
    │                       │  _extract(QualityEvaluation)                    │
    │                       │────────────────────────►│                       │
    │                       │  QualityEvaluation      │                       │
    │                       │◄────────────────────────│                       │
    │                       │                         │                       │
    │                       │  _should_refine() → "refine" or "done"          │
    │                       │────────────────────────────────────────────────►│
    │                       │                         │                       │
    │                       │  [conditional: refine loop or format]           │
    │                       │                         │                       │
    │                       │  final_state            │                       │
    │                       │◄────────────────────────────────────────────────│
    │                       │                         │                       │
    │  DeepSummarizerOutput │                         │                       │
    │◄──────────────────────│                         │                       │
```

### Execution Flow: With Lifecycle Layer

```
-User code        AgentExecutor         Lifecycle        Agent          Extraction
    │                 │                    │               │                │
    │  executor.run() │                    │               │                │
    │────────────────►│                    │               │                │
    │                 │  validate input    │               │                │
    │                 │───────────────────►│               │                │
    │                 │  check budget      │               │                │
    │                 │───────────────────►│               │                │
    │                 │  start trace span  │               │                │
    │                 │───────────────────►│               │                │
    │                 │                    │               │                │
    │                 │  agent.run(input, context)         │                │
    │                 │──────────────────────────────────►│                │
    │                 │                    │               │                │
    │                 │                    │  [per _extract() call]         │
    │                 │                    │               │  _extract()    │
    │                 │                    │               │───────────────►│
    │                 │                    │               │  result + cost │
    │                 │                    │               │◄───────────────│
    │                 │                    │               │                │
    │                 │                    │  accumulate step cost          │
    │                 │                    │◄──────────────│                │
    │                 │                    │  check step budget             │
    │                 │                    │  emit step span                │
    │                 │                    │               │                │
    │                 │                    │  [repeat for each step]        │
    │                 │                    │               │                │
    │                 │  output            │               │                │
    │                 │◄─────────────────────────────-─────│                │
    │                 │                    │               │                │
    │                 │  close trace span  │               │                │
    │                 │───────────────────►│               │                │
    │                 │  record in tracker │               │                │
    │                 │───────────────────►│               │                │
    │                 │                    │               │                │
    │  ExecutionResult│                    │               │                │
    │◄────────────────│                    │               │                │
    │  .output (typed)│                    │               │                │
    │  .metrics       │                    │               │                │
    │    .cost_usd    │                    │               │                │
    │    .steps[]     │                    │               │                │
```

### Execution Flow: MCP

```
MCP Client          FastMCP           MCP Tool Handler        Agent
(Claude, Cursor)       │                    │                    │
    │                  │                    │                    │
    │  tool call       │                    │                    │
    │  (JSON params)   │                    │                    │
    │─────────────────►│                    │                    │
    │                  │  invoke handler    │                    │
    │                  │───────────────────►│                    │
    │                  │                    │                    │
    │                  │                    │  deserialize to    │
    │                  │                    │  Pydantic InputT   │
    │                  │                    │                    │
    │                  │                    │  agent.run(input)  │
    │                  │                    │───────────────────►│
    │                  │                    │                    │
    │                  │                    │  [full multi-step  │
    │                  │                    │   workflow executes│
    │                  │                    │   internally]      │
    │                  │                    │                    │
    │                  │                    │  OutputT           │
    │                  │                    │◄───────────────────│
    │                  │                    │                    │
    │                  │                    │  serialize to JSON │
    │                  │  JSON response     │                    │
    │                  │◄───────────────────│                    │
    │  tool result     │                    │                    │
    │  (JSON)          │                    │                    │
    │◄─────────────────│                    │                    │
```

The MCP client has no visibility into the internal multi-step workflow. It sent one tool call and received one result. The 7 internal steps, the self-evaluation, the refinement loops — all invisible.

---

## Package Structure

```
dopeagents/
├── __init__.py                    # Public API re-exports
├── py.typed                       # PEP 561 type marker
│
├── core/                          # Agent Layer
│   ├── __init__.py
│   ├── agent.py                   # Agent[InputT, OutputT] base class
│   │                              #   - _extract() / _extract_partial()
│   │                              #   - run() abstract method
│   │                              #   - debug() / describe()
│   │                              #   - as_mcp_tool() / as_mcp_server()
│   │                              #   - _build_graph() abstract method
│   ├── context.py                 # AgentContext (model override, metadata, budget)
│   ├── state.py                   # Internal graph state schemas (TypedDict)
│   └── types.py                   # Shared type definitions
│
├── agents/                        # Concrete agent implementations
│   ├── __init__.py                # Re-exports all agents + their I/O types
│   ├── deep_summarizer.py         # DeepSummarizer + Input/Output models
│   ├── research_agent.py          # ResearchAgent + Input/Output models
│   └── ...                        # Future agents
│
├── contracts/                     # Typed composition
│   ├── __init__.py
│   ├── checker.py                 # ContractChecker — verify agent compatibility
│   └── pipeline.py                # Pipeline — validated multi-agent sequences
│
├── lifecycle/                     # Lifecycle Layer
│   ├── __init__.py
│   ├── executor.py                # AgentExecutor — wraps run() with lifecycle
│   └── result.py                  # AgentExecutionResult, ExecutionMetrics, StepMetrics
│
├── cost/                          # Cost tracking and budget enforcement
│   ├── __init__.py
│   ├── tracker.py                 # CostTracker — accumulates per-step, per-agent, global
│   └── budget.py                  # BudgetConfig — limits at multiple granularities
│
├── resilience/                    # Retry, fallback, degradation
│   ├── __init__.py
│   ├── retry.py                   # RetryPolicy — step-level retry with backoff
│   └── degradation.py            # DegradationChain — agent-level fallback
│
├── observability/                 # Tracing and debugging
│   ├── __init__.py
│   ├── tracer.py                  # Tracer protocol + ConsoleTracer
│   ├── otel.py                    # OTelTracer — OpenTelemetry integration
│   └── debug.py                   # DebugInfo — graph + prompts + schemas without execution
│
├── adapters/                      # Wrapping external code
│   ├── __init__.py
│   └── wrap.py                    # wrap_function(), wrap_class()
│
├── mcp_server/                    # MCP exposure
│   ├── __init__.py
│   ├── server.py                  # create_mcp_server() factory
│   └── registry.py               # Agent catalog as MCP resource
│
└── cli/                           # CLI entry points
    ├── __init__.py
    └── main.py                    # dopeagents list|run|dry-run|describe|mcp
```

### Module Dependency Rules

Dependencies flow strictly downward through the layers. No upward or circular dependencies are permitted.

```
cli/              → lifecycle/, agents/, mcp_server/
mcp_server/       → core/, agents/
adapters/         → core/
lifecycle/        → core/, cost/, resilience/, observability/
contracts/        → core/
agents/           → core/
core/             → (external: instructor, litellm, langgraph, pydantic)
cost/             → (standalone, no internal deps)
resilience/       → (standalone, no internal deps)
observability/    → (standalone, no internal deps)
```

Key constraints:

- `core/agent.py` imports `instructor` and `litellm` (Extraction Layer)
- `core/agent.py` imports `langgraph` (Orchestration Layer) — but only inside `_build_graph()`
- `agents/*` import only from `core/`
- `lifecycle/` imports from `core/` but never from `agents/` — it works with the `Agent` base class, not concrete implementations
- `cost/`, `resilience/`, `observability/` have no internal dependencies — they can be used standalone

---

## The Agent Base Class

The `Agent[InputT, OutputT]` generic base class is the central abstraction. It defines the contract that all agents — built-in and user-defined — must satisfy.

```python
class Agent(Generic[InputT, OutputT]):
    """
    Base class for all DopeAgents agents.
  
    Subclasses must:
    1. Define InputT and OutputT as Pydantic models
    2. Declare class metadata (name, version, description, etc.)
    3. Declare step_prompts as a ClassVar[dict[str, str]]
    4. Implement _build_graph() → compiled LangGraph
    5. Implement run(InputT) → OutputT
    6. Implement step methods called by the graph
    """
  
    # --- Class metadata (declared by subclasses) ---
    name: ClassVar[str]
    version: ClassVar[str]
    description: ClassVar[str]
    capabilities: ClassVar[list[str]]
    tags: ClassVar[list[str]]
    requires_llm: ClassVar[bool]
    default_model: ClassVar[str]
    step_prompts: ClassVar[dict[str, str]]
  
    # --- Instance configuration ---
    model: str                    # LiteLLM model string
    step_models: dict[str, str]   # per-step model overrides
    system_prompt: str | None     # global system prompt override
  
    # --- Provided by base class ---
  
    def _extract(self, response_model: type[T], messages: list[dict], **kwargs) -> T:
        """
        Structured LLM extraction via Instructor + LiteLLM.
        Every LLM interaction in every step goes through this method.
        """
        ...
  
    def _extract_partial(self, response_model: type[T], messages: list[dict], **kwargs) -> Iterator[T]:
        """Streaming variant of _extract() for real-time UIs."""
        ...
  
    def run(self, input: InputT, context: AgentContext | None = None) -> OutputT:
        """Execute the agent. Subclasses implement this."""
        ...
  
    def debug(self, input: InputT) -> DebugInfo:
        """
        Return graph structure, step prompts, and step schemas
        without executing anything. No API calls.
        """
        ...
  
    def describe(self) -> AgentDescription:
        """Return agent metadata for discovery and composition."""
        ...
  
    def as_mcp_tool(self, mcp: FastMCP) -> None:
        """Register this agent as an MCP tool on the given server."""
        ...
  
    def as_mcp_server(self, name: str | None = None) -> FastMCP:
        """Create a standalone MCP server for this agent."""
        ...
  
    def _build_graph(self):
        """Construct the internal LangGraph. Subclasses implement this."""
        ...
```

### The `_extract()` Method

This is the single most important method in the architecture. It is the only point where agents interact with LLMs, and it is where the Extraction Layer connects to the Agent Layer.

```python
def _extract(self, response_model: type[T], messages: list[dict], **kwargs) -> T:
    # 1. Determine which model to use for the current step
    model = self._resolve_model_for_current_step()
  
    # 2. Get or create the Instructor client
    client = self._get_instructor_client()
  
    # 3. Call Instructor, which calls LiteLLM, which calls the provider
    response = client.chat.completions.create(
        model=model,
        response_model=response_model,
        messages=messages,
        **kwargs
    )
  
    # 4. Instructor handles schema validation and re-ask automatically
    # 5. LiteLLM provides response_cost and token metadata automatically
    # 6. The Lifecycle Layer hooks into this method to capture step metrics
  
    return response
```

The Lifecycle Layer instruments `_extract()` — not by modifying it, but by hooking into Instructor's event system and LiteLLM's response metadata. This keeps `_extract()` simple and the agent's step methods unaware of lifecycle concerns.

---

## Cost Tracking Architecture

Cost tracking is designed around a simple principle: **cost data originates at the extraction point and aggregates upward**.

```
_extract() call
    │
    ├── LiteLLM response includes: response_cost, prompt_tokens, completion_tokens
    │
    ▼
StepMetrics (per step)
    │
    ├── step_name, cost_usd, tokens, latency_ms, model
    │
    ▼
ExecutionMetrics (per agent run)
    │
    ├── total_cost_usd, total_latency_ms, steps: list[StepMetrics]
    │
    ▼
CostTracker (session-wide)
    │
    ├── total_cost_usd
    ├── by_agent: {agent_name: {cost, calls, avg_cost, by_step: {...}}}
    │
    ▼
BudgetConfig (enforcement)
    │
    ├── max_cost_per_step   → checked after each _extract()
    ├── max_cost_per_call   → checked after each step completes
    ├── max_cost_global     → checked before each agent run
    └── on_exceeded         → "error" | "degrade"
```

### Budget Enforcement Points

Budget is checked at three points during execution:

1. **Before agent run:** Is the global budget already exhausted? If so, don't start.
2. **After each `_extract()` call:** Has this step exceeded `max_cost_per_step`? If so, either error or degrade.
3. **After each graph step completes:** Has the total run exceeded `max_cost_per_call`? If so, either error or return best-so-far.

The `on_exceeded="degrade"` mode is critical for agents with refinement loops. If budget is hit during refinement round 3, the agent returns the output from refinement round 2 — the best result produced within budget — rather than discarding all work.

---

## Retry Architecture

Retry operates at three distinct levels, each handling different failure modes:

```
Level 1: Extraction (Instructor)
├── Handles: Schema validation failures (missing fields, wrong types)
├── Mechanism: Instructor re-asks the LLM with validation error in prompt
├── Scope: Single _extract() call
├── Transparent to: Agent step methods (they never see the retry)
│
Level 2: Step (DopeAgents Lifecycle Layer)
├── Handles: Infrastructure failures (timeouts, rate limits, transient errors)
├── Mechanism: RetryPolicy with exponential backoff
├── Scope: Single graph step (not the entire agent run)
├── Key property: Only the failed step retries; completed steps are preserved
│
Level 3: Agent (DegradationChain)
├── Handles: Fundamental failures (model down, context too long, persistent errors)
├── Mechanism: Try agents in order until one succeeds
├── Scope: Entire agent — falls through to simpler alternatives
├── Key property: Last agent in chain should always succeed (e.g., no-LLM fallback)
```

### Step-Level Retry Boundary

The most architecturally significant decision in the retry system is that **step-level retry re-executes only the failed step, not the entire agent**. This is possible because LangGraph maintains graph state across steps. When step 5 fails and retries, steps 1–4 are not re-executed — their results are preserved in the graph state.

This is especially important for expensive agents. A DeepSummarizer run involves 5–8 LLM calls. Re-running the entire agent on a step 5 failure would double the cost. Re-running only step 5 adds marginal cost.

---

## MCP Integration Architecture

MCP integration maps DopeAgents concepts onto MCP primitives:

```
DopeAgents                          MCP Protocol
─────────────────────────────────────────────────────
Agent.run()                    →    Tool (callable)
Agent InputT/OutputT schemas   →    Tool JSON Schema (auto-generated)
Agent.describe()               →    Tool metadata
Agent.debug().step_prompts     →    Prompt (inspectable)
Agent catalog                  →    Resource (dopeagents://catalog)
```

### Schema Generation

MCP tool schemas are generated automatically from the agent's Pydantic input model. No manual schema definition is needed:

```python
# DeepSummarizerInput (Pydantic model)
class DeepSummarizerInput(BaseModel):
    text: str = Field(..., min_length=1, description="Text to summarize")
    max_length: int = Field(default=500, ge=10, le=10000)
    style: Literal["paragraph", "bullets", "tldr"] = "paragraph"

# Automatically becomes MCP tool schema:
# {
#   "name": "DeepSummarizer",
#   "description": "Handles arbitrarily long documents with chunking...",
#   "inputSchema": {
#     "type": "object",
#     "properties": {
#       "text": {"type": "string", "minLength": 1, "description": "Text to summarize"},
#       "max_length": {"type": "integer", "default": 500, "minimum": 10, "maximum": 10000},
#       "style": {"type": "string", "enum": ["paragraph", "bullets", "tldr"], "default": "paragraph"}
#     },
#     "required": ["text"]
#   }
# }
```

### Transport Modes

MCP servers support two transport modes:

- **stdio:** For local integration with Claude Desktop, Cursor, and similar tools. The MCP server runs as a subprocess, communicating over stdin/stdout.
- **Streamable HTTP:** For remote/networked deployments. The MCP server runs as an HTTP service on a configurable port.

Both transports expose the exact same tools, prompts, and resources. The transport is a deployment concern, not an agent concern.

---

## Composition Architecture

### Contract Checking

The `ContractChecker` performs static analysis on agent input/output Pydantic schemas to verify compatibility before execution:

```
ContractChecker.check(SourceAgent, TargetAgent, field_mappings)
    │
    ├── Introspect SourceAgent.OutputT schema (field names, types)
    ├── Introspect TargetAgent.InputT schema (field names, types, required/optional)
    │
    ├── For each required field in TargetAgent.InputT:
    │   ├── Check if field name exists in SourceAgent.OutputT with compatible type
    │   ├── Check if field name has explicit mapping in field_mappings
    │   └── If neither → incompatible (error with specific field info)
    │
    ├── For each optional field in TargetAgent.InputT:
    │   └── If no match and no mapping → warning (will use default)
    │
    └── Return CheckResult(compatible, field_matches, warnings, errors)
```

### Pipeline Validation

The `Pipeline` class extends contract checking to multi-step sequences:

```
Pipeline([AgentA, AgentB, AgentC], field_mappings={...})
    │
    ├── Check AgentA.OutputT → AgentB.InputT compatibility
    ├── Check AgentB.OutputT → AgentC.InputT compatibility
    │
    ├── If any step is incompatible:
    │   └── PipelineValidationError with step index and specific field info
    │
    └── If all steps compatible:
        └── Pipeline is valid — can be executed with confidence
```

Pipeline validation happens at construction time. No LLM calls are made. No tokens are burned. Incompatible pipelines fail immediately with actionable error messages.

---

## Observability Architecture

Observability is layered and opt-in at each level:

```
Layer 1: Debug Mode (no execution)
├── Agent.debug(input) → DebugInfo
├── Contains: graph structure, step prompts, step schemas, message templates
├── No API calls, no cost, no side effects
├── Use case: Inspect what an agent will do before running it
│
Layer 2: Step-Level Metrics (always captured)
├── Every _extract() call records: cost, tokens, latency, model, step name
├── Available in ExecutionMetrics.steps after run completes
├── No additional configuration needed
│
Layer 3: Console Tracing (opt-in)
├── ConsoleTracer prints step-level events to stderr
├── Configured via AgentExecutor(tracer=ConsoleTracer())
│
Layer 4: OpenTelemetry (opt-in)
├── OTelTracer emits spans compatible with any OTel collector
├── Parent span per agent run, child span per internal step
├── Cost, tokens, latency attached as span attributes
├── Configured via AgentExecutor(tracer=OTelTracer())
```

### PII Redaction

Fields in Pydantic schemas can be marked as containing PII:

```python
class SensitiveInput(BaseModel):
    document: str = Field(..., json_schema_extra={"pii": True})
    metadata: dict
```

The observability layer checks for `pii: True` in field schema extras. Marked fields flow through agent logic normally (the agent processes the actual data) but are redacted in all observability output — trace spans, console logs, debug info, and cost tracker summaries.

---

## Extension Points

### Adding a New Agent

1. Define Pydantic input/output models
2. Subclass `Agent[InputT, OutputT]`
3. Declare class metadata and `step_prompts`
4. Implement `_build_graph()` with a LangGraph `StateGraph`
5. Implement step methods that call `_extract()`
6. Implement `run()` that invokes the graph and maps state to output

The agent automatically gets: cost tracking, budget guards, retry, observability, debug mode, MCP exposure, composition checks, and CLI support. None of these require any code in the agent itself.

### Customizing an Existing Agent

Override individual step methods or `step_prompts` via subclassing:

```python
class LegalSummarizer(DeepSummarizer):
    step_prompts = {
        **DeepSummarizer.step_prompts,
        "evaluate": "You are a legal summary evaluator. Score on citation preservation...",
    }
```

The full graph, all other steps, and all infrastructure are inherited automatically.

### Wrapping External Code

Use `wrap_function()` or `wrap_class()` from `dopeagents.adapters.wrap` to give existing code full DopeAgents capabilities. The only requirement is Pydantic models for input and output.

### Adding a New Lifecycle Concern

Implement a new component (e.g., rate limiter, A/B test router) and integrate it into `AgentExecutor`. No agent code changes. The new concern wraps agent execution from the outside, just like cost tracking and retry do today.

### Adding a New Tracer

Implement the `Tracer` protocol:

```python
class Tracer(Protocol):
    def on_agent_start(self, agent: Agent, input: BaseModel) -> None: ...
    def on_step_start(self, step_name: str) -> None: ...
    def on_step_end(self, step_name: str, metrics: StepMetrics) -> None: ...
    def on_agent_end(self, result: AgentExecutionResult) -> None: ...
```

Pass the implementation to `AgentExecutor(tracer=MyTracer())`. All agents automatically emit events to the new tracer.

---

## Design Constraints and Trade-offs

### LangGraph as Internal Dependency

**Decision:** LangGraph is used for internal graph orchestration but is never exposed in the public API.

**Trade-off:** Users cannot leverage LangGraph's advanced features (human-in-the-loop, checkpointing, LangSmith integration) through DopeAgents. If they need those features, they should use LangGraph directly.

**Rationale:** Exposing the graph engine would couple the public API to LangGraph's API. If LangGraph's API changes, or if a better orchestration engine emerges, migration would break all user code. By keeping it internal, we preserve the ability to swap orchestration engines without breaking any external interface.

### Instructor as Extraction Layer

**Decision:** All structured LLM output goes through Instructor's `_extract()` pattern.

**Trade-off:** Agents cannot use raw unstructured LLM output. Every LLM interaction must define a Pydantic response model.

**Rationale:** Structured output is a prerequisite for typed composition, step-level metrics, and reliable multi-step workflows. Unstructured output in a multi-step graph leads to fragile string parsing between steps. The constraint of always using structured output makes agents composable and debuggable.

### Pydantic-First Contracts

**Decision:** Every agent must define Pydantic models for input and output.

**Trade-off:** Wrapping a simple function requires defining two Pydantic models, which can feel heavyweight for trivial use cases.

**Rationale:** Pydantic models are the foundation that enables everything else — MCP schema generation, composition checking, input validation, debug mode, observability, and caching. Without typed contracts, none of these features would work. The overhead of defining models is small relative to the capabilities they unlock.

### No Agent-to-Agent Communication During Execution

**Decision:** Agents in a pipeline execute sequentially. There is no mechanism for agents to communicate during execution (e.g., agent B querying agent A while agent A is still running).

**Trade-off:** Cannot express concurrent or collaborative multi-agent patterns within DopeAgents itself.

**Rationale:** Sequential composition with typed contracts covers the vast majority of production use cases. Concurrent multi-agent communication introduces significant complexity (deadlocks, race conditions, shared state management) that is better handled by dedicated orchestration tools when genuinely needed. DopeAgents focuses on making individual agents excellent and their sequential composition safe.
