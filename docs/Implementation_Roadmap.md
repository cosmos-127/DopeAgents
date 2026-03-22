# DopeAgents — Implementation Roadmap (Refined)

> **Roadmap Version**: 2.1.0
> **Tracks**: Design Document v3.0.0
> **Status**: Living document — update task status as work lands
> **Last Updated**: March 22, 2026

---

## Status Summary

| Phase | Name                        | Status      | Progress                                    |
| ----- | --------------------------- | ----------- | ------------------------------------------- |
| 0     | Foundation                  | ✅ COMPLETE | All core types, Agent base, errors defined |
| 1     | DeepSummarizer (7-step)     | ✅ COMPLETE | Full workflow, step_prompts customizable    |
| 2     | Infrastructure Layer        | ✅ COMPLETE | Cost tracking, budget guards, caching, resilience |
| 3     | ResearchAgent + Composition | ✅ COMPLETE | 6-step agent, step_prompts customizable    |
| 4     | MCP Exposure                | ⏳ NOT STARTED | FastMCP server and tool registration       |
| 5     | CLI, Sandbox & Polish       | ⏳ NOT STARTED | CLI commands, interactive REPL, wrapping   |

### Key Milestones Achieved

- ✅ **Design Decision (DD-035)**: Implemented immutable `system_prompt` (ClassVar only) + customizable `step_prompts` (init-time, instance-level)
- ✅ **Pattern Standardization**: Both DeepSummarizer and ResearchAgent use identical `step_prompts` customization in `__init__`
- ✅ **Documentation Sync**: Design_Document.md §3.6 updated with implementation details, code examples, and design rationale

### Architectural Decision: Step Prompts Customization (DD-035)

**Choice**: Immutable `system_prompt` (ClassVar only) + Customizable `step_prompts` (init-time, instance-level)

**Rationale**:
- **`system_prompt`**: Agent identity — fixed at class definition time (P2: Contracts enforceable, P8: Stateless)
- **`step_prompts`**: Per-step guidance — mergeable at instance creation time via `**kwargs` extraction

**Pattern Used in Both Agents**:
```python
# In DeepSummarizer.__init__ and ResearchAgent.__init__:
def __init__(self, **kwargs: Any) -> None:
    custom_step_prompts = kwargs.pop("step_prompts", None)
    super().__init__(**kwargs)
    if custom_step_prompts is not None:
        self.step_prompts = {**self.step_prompts, **custom_step_prompts}
```

**Usage Example**:
```python
# Default step_prompts from ClassVar
agent1 = DeepSummarizer()

# Custom step_prompts merged with defaults
agent2 = DeepSummarizer(step_prompts={"analyze": "Focus on technical terms..."})

# Both instances are independent; behavior fixed at instance creation
agent1.run(input1)  # Uses default prompts
agent2.run(input2)  # Uses merged custom + default prompts
```

**Design Properties Preserved**:
- ✅ P1 (Externally simple): Single `step_prompts` dict parameter
- ✅ P2 (Contracts enforceable): Agent behavior locked at class/instance definition time
- ✅ P8 (Stateless): Same agent instance + same input = deterministic output; different instances have different step_prompts
- ✅ P10 (Transparency): All step_prompts inspectable via `describe()`, observability spans

**Where Implemented**:
- [dopeagents/agents/deep_summarizer.py](dopeagents/agents/deep_summarizer.py#L262) — Lines 262–280
- [dopeagents/agents/research_agent.py](dopeagents/agents/research_agent.py#L187) — Lines 187–202
- [docs/Design_Document.md](docs/Design_Document.md#L1136) — §3.6 with code examples and design rationale

---

## Overview

Six sequential phases, each independently releasable and testable.
All code lives under the `dopeagents/` package. All imports use `dopeagents.*`.

```text
Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 3
 ✅       ✅       ✅       ✅
           │                       │
           ├──────────────▶ Phase 4 (MCP)
           🔄                    ⏳
                                    │
                               Phase 5 (CLI & Polish)
                                    ⏳
```

| Phase | What                        | Scope  | Depends On  | Status       |
| ----- | --------------------------- | ------ | ----------- | ------------ |
| 0     | Foundation                  | Small  | Nothing     | ✅ COMPLETE  |
| 1     | DeepSummarizer (7-step)     | Large  | Phase 0     | ✅ COMPLETE  |
| 2     | Infrastructure Layer        | Large  | Phase 0 + 1 | ✅ COMPLETE  |
| 3     | ResearchAgent + Composition | Large  | Phase 0 + 2 | ✅ COMPLETE  |
| 4     | MCP Exposure                | Medium | Phase 0 + 1 | ⏳ NOT STARTED |
| 5     | CLI, Sandbox & Polish       | Medium | All above   | ⏳ NOT STARTED |

---

## Phase 0: Foundation

**What**: Project scaffolding, base classes, and core type system. Everything that every subsequent phase builds on. No agents, no LLM calls — just the skeleton and the `_extract()` primitive.

**Depends on**: Nothing.

**Estimated scope**: Small (1–2 days)

**Status**: ✅ **COMPLETE**

---

### Tasks (in order)

#### T0.1 — Project scaffolding and `pyproject.toml` configuration

**Project structure**: Create the complete directory tree with stub files:

- Root directory: `dopeagents/` (package root)
- Create all subdirectories: `core/`, `sandbox/`, `contracts/`, `lifecycle/`, `observability/`, `cost/`, `resilience/`, `cache/`, `adapters/`, `spec/`, `registry/`, `benchmark/`, `tools/`, `agents/`, `security/`
- For each directory, create `__init__.py` with appropriate re-exports (empty for now; detailed in T0.4–T0.6)
- Create `dopeagents/py.typed` (empty file; PEP 561 marker)
- Create stub `.py` files for each module per the directory tree (e.g., `dopeagents/core/agent.py`, `dopeagents/agents/deep_summarizer.py`, `dopeagents/agents/research_agent.py`, etc.); stub content is a docstring and `pass` for now
- Create `tests/` directory structure mirroring `dopeagents/` (e.g., `tests/core/`, `tests/agents/`, etc.)
- Create `.gitignore` with entries for `.venv`, `__pycache__`, `.pytest_cache`, `*.egg-info`, `dist/`, `build/`, `.mypy_cache`, `ruff_cache/`, `.ruff_cache`, `.cache`, `*.db`

**pyproject.toml** configuration:

- **Package manager**: `uv` — use `uv add` for dependency management; lock file is `uv.lock`
- **Project metadata**:
  - `name = "dopeagents"`
  - `version = "0.1.0"`
  - `description = "Agentic AI framework with structured contracts, cost tracking, and multi-framework support"`
  - `authors = [...]`, `license = {...}`, `readme = "README.md"`
- Declare all core dependencies with version pins (DD §2.2):
  - `pydantic >= 2.11`
  - `instructor >= 1.14`
  - `litellm >= 1.56`
  - `langgraph >= 0.2`
  - `typing-extensions >= 4.12`
  - `click >= 8.1`
  - `jsonschema >= 4.23`
  - `httpx >= 0.27`
  - `packaging >= 21.0` (required by Registry version parsing — DD §14)
- Declare optional dependency groups:
  - `mcp`: `fastmcp >= 3.0, < 4.0`
  - `cache`: `diskcache >= 5.6`
  - `otel`: `opentelemetry-api`, `opentelemetry-sdk`
  - `langchain`: `langchain-core >= 0.2`
  - `langgraph-adapter`: `langgraph >= 0.2` (separate from core langgraph if needed)
  - `crewai`: `crewai >= 0.30`
  - `autogen`: `autogen-agentchat >= 0.2`
- Configure `[project.scripts]` entry point: `dopeagents = dopeagents.cli:main`
- Configure `[tool.pytest.ini_options]` with `testpaths = ["tests"]`
- **Linter**: configure `ruff` under `[tool.ruff]`:
  - `line-length = 100`
  - `target-version = "py311"`
  - `select = ["E", "F", "I", "UP", "B", "SIM"]` (errors, undefined names, imports, pyupgrade, bugbear, simplify)
  - Enable `[tool.ruff.lint.isort]` for import sorting
- **Type checker**: configure `mypy` strict mode under `[tool.mypy]`:
  - `strict = true`
  - `python_version = "3.11"`
  - `ignore_missing_imports = true`
  - `disallow_untyped_defs = true`
- Add `ruff`, `mypy`, `pytest`, `pytest-cov` to `[dependency-groups.dev]` (installed via `uv sync --group dev`)
- Configure `[tool.uv]` with `python = "^3.11"`

#### T0.2 — `dopeagents/py.typed` marker

- File: `dopeagents/py.typed`
- Empty file. Declares the package is PEP 561 typed.

#### T0.3 — `dopeagents/errors.py` — full error hierarchy

- File: `dopeagents/errors.py`
- Implement the complete typed error hierarchy from DD §19:
  - `DopeAgentsError` (base)
  - Contract errors: `ContractError`, `PipelineValidationError`, `InputValidationError`, `OutputValidationError`
  - Execution errors: `ExecutionError`, `AgentExecutionError`, `AllFallbacksFailedError`
  - Cost errors: `CostError`, `BudgetExceededError`, `BudgetDegradedError`
  - Registry errors: `RegistryError`, `AgentNotFoundError`, `AgentValidationError`
  - Tool errors: `ToolError`, `ToolExecutionError`
  - Adapter errors: `AdapterError`, `FrameworkNotInstalledError`
  - MCP errors: `MCPError`, `MCPNotInstalledError`, `MCPRegistrationError`, `MCPServerError`
  - Extraction errors: `ExtractionError`, `ExtractionValidationError`, `ExtractionProviderError`
- All errors carry structured fields (agent_name, original_error, etc.) — no bare string messages.

> **Refinement note**: `BudgetDegradedError` is referenced in DD §8.2 (Budget Guard) but missing from the §19 error taxonomy code block. Include it here under `CostError` alongside `BudgetExceededError`. It must carry `agent_name: str`, `current_cost: float`, `budget_limit: float`, `degraded_output: Any`.

#### T0.4 — `dopeagents/core/context.py` — `AgentContext`

- File: `dopeagents/core/context.py`
- Implement `AgentContext` (DD §3.3):
  - `run_id: UUID` (auto-generated)
  - `trace_id: UUID | None`
  - `parent_agent: str | None`
  - `timestamp: datetime`
  - `environment: str = "development"`
  - `max_cost_usd: float | None`
  - `max_tokens: int | None`
  - `model_override: str | None`
  - `mcp_request_id: str | None`
  - `metadata: dict[str, Any]`

#### T0.5 — `dopeagents/core/types.py` — result and metrics models

- File: `dopeagents/core/types.py`
- Implement three Pydantic models (DD §3.4):
  - `StepMetrics` — per-step cost, latency, token counts for multi-step agents
  - `ExecutionMetrics` — aggregate metrics including step breakdown
  - `AgentResult[OutputT]` — generic output wrapper with run metadata

#### T0.6 — `dopeagents/core/metadata.py` — `AgentMetadata`

- File: `dopeagents/core/metadata.py`
- Implement `AgentMetadata` (DD §3.2): name, version, description, capabilities, tags, requires_llm, default_model, system_prompt, step_prompts, input_schema, output_schema.

#### T0.7 — `dopeagents/core/agent.py` — `Agent` base class

- File: `dopeagents/core/agent.py`
- Implement the full `Agent[InputT, OutputT]` base class (DD §3.1):
  - Class-level metadata: `name`, `version`, `description`, `capabilities`, `tags`, `requires_llm`, `default_model`
  - Prompt declarations: `system_prompt: ClassVar[str]`, `step_prompts: ClassVar[dict[str, str]]`
  - `__init__`: accepts `model`, `system_prompt` override, `step_models`
  - `__init_subclass__`: validates system_prompt type, warns if requires_llm=True but no prompts declared
  - `_get_client()`: lazy-initializes Instructor client over LiteLLM (thread-safe via `threading.Lock`)
  - `_get_graph()`: lazy-builds the internal LangGraph (calls `_build_graph()`)
  - `_build_graph()`: default returns `None` (single-step agents skip this)
  - `_extract()`: core LLM call primitive via Instructor + LiteLLM
  - `_extract_partial()`: streaming extraction
  - `_model_for_step(step_name)`: resolves per-step model overrides
  - `input_type()` / `output_type()`: type introspection via `__orig_bases__`
  - `metadata()`: returns `AgentMetadata`
  - `to_metadata()`: returns instance-level metadata dict
  - `run()`: abstract method
  - `debug()` → `DebugInfo`
  - `describe()` → `AgentDescription`
  - `_has_loops()`: returns `False` by default
  - Framework adapter stubs: `as_langchain_runnable()`, `as_langgraph_node()`, `as_crewai_tool()`, `as_autogen_function()`, `as_openai_function()`, `as_callable()`, `as_mcp_tool()`, `as_mcp_server()`
- Implement `DebugInfo` and `AgentDescription` models in the same file.

> **Refinement note**: Added `as_langgraph_node()` to the adapter stubs list. DD §11.2 defines a full LangGraph adapter (`to_langgraph_node()`) that was missing from the original roadmap. The stub delegates to `dopeagents.adapters.langgraph.to_langgraph_node()`.

#### T0.8 — `dopeagents/config.py` — `DopeAgentsConfig`

- File: `dopeagents/config.py`
- Implement `DopeAgentsConfig` (DD §20) with:
  - `from_env()`: reads `DOPEAGENTS_*` environment variables
  - `from_toml()`: reads `dopeagents.toml` if present
  - `get_config()` / `set_config()`: global singleton accessors

> **Refinement note**: Environment variable prefix must be `DOPEAGENTS_*` (not `AGENTMAKER_*`). The class name is `DopeAgentsConfig`. The TOML file is `dopeagents.toml`.

#### T0.9 — `dopeagents/__init__.py` — minimal re-exports

- File: `dopeagents/__init__.py`
- Expose `__version__`, `Agent`, `AgentContext`, `AgentResult`, `ExecutionMetrics`, `StepMetrics`
- Do NOT re-export agents or deep submodules yet (they don't exist)

> **Refinement note**: `ExecutionMetrics` and `StepMetrics` must be included in the initial re-exports (they were missing in the original roadmap's "Definition of Done" but are part of the public API per DD §18.1).

#### T0.10 — Foundation tests

- Files: `tests/test_errors.py`, `tests/test_config.py`, `tests/core/test_agent.py`
- Verify `InputT`/`OutputT` type resolution works for a concrete `Agent` subclass
- Verify `_extract()` raises a typed error (not a bare exception) when Instructor fails
- Verify `AgentContext` auto-generates `run_id`
- Verify config loads from environment variables with `DOPEAGENTS_*` prefix
- Verify `__orig_bases__` introspection works for a two-level subclass hierarchy

---

### Definition of Done — Phase 0

- [X] `python -c "from dopeagents import Agent, AgentContext, AgentResult, ExecutionMetrics, StepMetrics"` succeeds
- [X] A minimal concrete `Agent[SomeInput, SomeOutput]` subclass can be defined with `name`, `version`, `description`, `capabilities` and a `run()` implementation
- [X] `Agent.input_type()` and `Agent.output_type()` resolve correctly (including two-level subclass hierarchies)
- [X] `agent._extract()` is callable and calls Instructor (`instructor.from_litellm`) — confirmed by monkeypatching the client in a test
- [X] All errors in `errors.py` are importable and carry their structured fields (including `BudgetDegradedError`)
- [X] `DopeAgentsConfig.from_env()` reads `DOPEAGENTS_*` env vars correctly
- [X] `mypy --strict dopeagents/core/` exits 0
- [X] `pytest tests/` passes (all foundation tests green)

**Status**: ✅ **COMPLETE** (Verified Mar 22, 2026)

---

### Risks / Unknowns — Phase 0

| Risk                                                                     | Mitigation                                                                         |
| ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `instructor.from_litellm` API may differ across instructor versions    | Pin `instructor >= 1.14` and test the `_get_client()` lazy init against a mock |
| `__orig_bases__` introspection is fragile under deep inheritance       | Add an explicit test with a two-level subclass hierarchy                           |
| LiteLLM `_hidden_params.response_cost` availability varies by provider | Default to `0.0` in `_extract()` and document that cost is best-effort         |
| `_get_client()` lazy init is not thread-safe                           | Use `threading.Lock` to guard lazy initialization of the Instructor client       |

---

## Phase 1: First Agent — DeepSummarizer

**What**: The flagship agent. A 7-step multi-step LangGraph workflow with chunking, multi-pass synthesis, self-evaluation, and iterative refinement. This is the proof that the architecture works end-to-end.

**Depends on**: Phase 0 complete (Agent base, `_extract()`, types).

**Estimated scope**: Large (3–5 days)

**Status**: ✅ **COMPLETE**

---

### Tasks (in order)

#### T1.1 — Design the internal LangGraph state

- File: `dopeagents/agents/deep_summarizer.py`
- Define `DeepSummarizerState` as a `TypedDict` with all state keys shared across steps:
  ```python
  class DeepSummarizerState(TypedDict):
      text: str
      max_length: int
      style: str
      focus: str | None
      analysis: dict          # output of analyze step
      chunks: list[str]       # output of chunk step
      chunk_summaries: list[str]  # output of summarize step
      synthesis: str          # output of synthesize step
      quality_score: float    # output of evaluate step
      feedback: str           # output of evaluate step
      refined: str            # output of refine step
      refinement_rounds: int
      max_refinement_loops: int  # guard variable (default 3)
      final_summary: str      # output of format step
      key_points: list[str]
      word_count: int
      truncated: bool
  ```
- Define all internal step output schemas (`_AnalyzeOut`, `_ChunkOut`, `_SummarizeOut`, `_SynthesizeOut`, `_EvaluateOut`, `_RefineOut`, `_FormatOut`) as private Pydantic models in the same file.

> **Refinement note**: Added `max_refinement_loops` as an explicit state key (default 3). The original roadmap mentioned it in T1.5 but did not include it in the state definition. This guard prevents unbounded refinement loops.

#### T1.2 — Implement `DeepSummarizerInput` and `DeepSummarizerOutput`

- File: `dopeagents/agents/deep_summarizer.py`
- `DeepSummarizerInput` (DD §17.1):
  - `text: str` (min_length=1)
  - `max_length: int = Field(default=500, ge=50, le=10000)`
  - `style: Literal["paragraph", "bullets", "tldr"] = "paragraph"`
  - `focus: str | None = None`
- `DeepSummarizerOutput` (DD §17.1):
  - `summary: str`
  - `key_points: list[str]`
  - `quality_score: float = Field(ge=0.0, le=1.0)`
  - `refinement_rounds: int`
  - `chunks_processed: int`
  - `word_count: int`
  - `truncated: bool`

#### T1.3 — Implement class-level metadata and step_prompts

- File: `dopeagents/agents/deep_summarizer.py`
- Declare `name`, `version`, `description`, `capabilities`, `tags` as `ClassVar`
- Declare `step_prompts: ClassVar[dict[str, str]]` for all 7 steps:
  - `analyze`: instruct the model to characterize text length, structure, and recommend a chunking strategy
  - `chunk`: split text into semantically coherent segments
  - `summarize`: summarize each chunk independently and extract its key points
  - `synthesize`: combine chunk summaries into a unified, coherent whole with the requested `style`
  - `evaluate`: score the synthesis from 0.0 to 1.0; identify weaknesses; provide actionable feedback
  - `refine`: rewrite the synthesis using the evaluation feedback to address its weaknesses
  - `format`: enforce `max_length`, apply `style` formatting, compute metrics, determine `truncated`
- Declare `step_models` alignment (DD §17.1):
  - **Fast/cheap (gpt-4o-mini)**: analyze, chunk, summarize, format
  - **Smart (gpt-4o or per config)**: synthesize, evaluate, refine

#### T1.4 — Implement all 7 step methods

- File: `dopeagents/agents/deep_summarizer.py`
- Each step method signature: `def _step_<name>(self, state: DeepSummarizerState) -> dict[str, Any]:`
- Each step calls `self._extract(response_model=_<Name>Out, messages=[...], model=self._model_for_step("<name>"))`
- `_step_analyze`: returns `analysis` dict with `recommended_chunk_size`, `text_type`, `complexity`
- `_step_chunk`: uses `analysis.recommended_chunk_size` to split text; returns `chunks: list[str]`
- `_step_summarize`: iterates over each chunk (multiple `_extract()` calls); returns `chunk_summaries: list[str]`
- `_step_synthesize`: combines summaries; returns `synthesis: str`, `key_points: list[str]`
- `_step_evaluate`: scores synthesis; returns `quality_score: float`, `feedback: str`
- `_step_refine`: improves synthesis using feedback; returns `refined: str`, increments `refinement_rounds`; **also updates `synthesis` with the refined text** so the next evaluate reads from it
- `_step_format`: applies style and length constraints; returns `final_summary`, `word_count`, `truncated`

> **Refinement note — `_step_summarize` chunk guard**: Add a `max_chunks` guard (default 10). If `len(chunks) > max_chunks`, truncate the chunk list before the summarize loop to prevent unbounded token cost. Log a warning when truncation occurs.

> **Refinement note — `_step_refine` state update**: The refine step must write back to `synthesis` (not only `refined`), because the evaluate step reads `synthesis`. Without this, the evaluate→refine→evaluate loop would re-evaluate the original pre-refine synthesis.

#### T1.5 — Implement `_build_graph()` with conditional edges

- File: `dopeagents/agents/deep_summarizer.py`
- Wire the LangGraph `StateGraph`:
  ```
  analyze → chunk → summarize → synthesize → evaluate
                                                │
                            quality_score < 0.8 AND refinement_rounds < max_refinement_loops
                                → refine → evaluate  (loop)
                            ELSE
                                → format → END
  ```
- Conditional edge function:
  ```python
  def _should_refine(state: DeepSummarizerState) -> str:
      if (state["quality_score"] < 0.8
          and state["refinement_rounds"] < state.get("max_refinement_loops", 3)):
          return "refine"
      return "format"
  ```
- Override `_has_loops()` to return `True`

> **Refinement note — Loop direction clarified**: DD §17.1 says "refine loops back to evaluate". The graph wiring is: `evaluate` has a conditional edge — if quality < 0.8 AND under max loops, go to `refine`; `refine` always transitions back to `evaluate` (unconditional edge). `evaluate` then re-checks and either loops again or exits to `format`. This means the loop is `evaluate → refine → evaluate`, NOT `refine → synthesize → evaluate`.

#### T1.6 — Implement `run()` for DeepSummarizer

- File: `dopeagents/agents/deep_summarizer.py`
- Call `self._get_graph().invoke(initial_state)` where initial state is built from `input`
- Map final state fields to `DeepSummarizerOutput`

#### T1.7 — Override `_render_prompt()` and `_get_model_config()`

- File: `dopeagents/agents/deep_summarizer.py`
- `_render_prompt()`: returns the formatted `step_prompts` as a rendered multi-step prompt overview (for `debug()`)
- `_get_model_config()`: returns dict with `model`, `step_models`, `max_retries`

#### T1.8 — `dopeagents/agents/__init__.py` re-exports

- File: `dopeagents/agents/__init__.py`
- Export `DeepSummarizer`, `DeepSummarizerInput`, `DeepSummarizerOutput`

#### T1.9 — Unit tests for DeepSummarizer

- File: `tests/agents/test_deep_summarizer.py`
- Test `DeepSummarizerInput` validation (min_length, style enum, max_length bounds)
- Test `DeepSummarizerOutput` field presence
- Test `describe()` returns correct `steps` list and `has_loops=True`
- Test `debug()` returns `is_multi_step=True` and `step_prompts` populated
- Test `_build_graph()` returns a compiled LangGraph (not None)
- Test `_model_for_step()` resolves step overrides correctly
- Test `max_chunks` guard truncates when chunk count exceeds 10
- All tests use mocked `_extract()` — no real LLM calls

#### T1.10 — Integration test: DeepSummarizer end-to-end

- File: `tests/integration/test_deep_summarizer_e2e.py`
- Requires a real API key (skipped with `pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="No API key")`)
- Call `DeepSummarizer().run(DeepSummarizerInput(text="...", style="bullets"))`
- Assert `output.summary` is non-empty
- Assert `output.quality_score >= 0.0` and `output.quality_score <= 1.0`
- Assert `output.refinement_rounds >= 0`
- Assert `output.chunks_processed >= 1`

> **Refinement note**: Use `pytest.mark.skipif` with an environment check, not bare `pytest.mark.skip`. The original roadmap said `pytest.mark.skip` which would unconditionally skip the test.

---

### Definition of Done — Phase 1

- [X] `DeepSummarizer().run(DeepSummarizerInput(text="...", style="bullets"))` returns a valid `DeepSummarizerOutput` with a real LLM
- [X] `DeepSummarizer().describe().steps` == `["analyze", "chunk", "summarize", "synthesize", "evaluate", "refine", "format"]`
- [X] `DeepSummarizer().describe().has_loops` is `True`
- [X] `DeepSummarizer().debug(input)` returns `is_multi_step=True` and populated `step_prompts`
- [X] A document with `quality_score < 0.8` on first synthesize triggers at least one `refine` loop
- [X] The refine loop respects `max_refinement_loops` (default 3) and terminates
- [X] `_step_summarize` respects `max_chunks` guard (default 10)
- [X] `mypy --strict dopeagents/agents/deep_summarizer.py` exits 0
- [X] All unit tests pass without a real API key (mocked `_extract()`)
- [X] **Step Prompts Customization**: `DeepSummarizer(step_prompts={"analyze": "custom..."})` merges with defaults and is instance-independent

**Status**: ✅ **COMPLETE** (Verified Mar 22, 2026)

**Implementation Notes**:
- Immutable `system_prompt` at ClassVar level (no runtime override accepted)
- Customizable `step_prompts` via init-time `**kwargs` extraction and merging
- Pattern: concrete agent's `__init__` calls `kwargs.pop("step_prompts", None)` before `super().__init__(**kwargs)`, then applies merge
- Each instance gets its own `self.step_prompts` dict shadowing the ClassVar

---

### Risks / Unknowns — Phase 1

| Risk                                                                                                             | Mitigation                                                                                       |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| LangGraph `StateGraph` API differences between `0.2.x` and later — `add_conditional_edges` shape may vary | Pin `langgraph >= 0.2, < 0.3` initially; test with the exact pinned version in CI              |
| `_step_summarize` iterates chunks = N calls to `_extract()`; token cost per run is unbounded                 | Add a `max_chunks` guard (default 10); truncate chunk list before the loop                     |
| Refinement loop diverges — quality never reaches 0.8 for some inputs                                            | `max_refinement_loops` guard in state (default 3); `degrade` path returns best result so far |
| Per-step model assignments are implicit in step_prompts defaults                                                 | Document step model strategy in class docstring; exposed via `describe()`                      |
| Refine step writes to `refined` but evaluate reads `synthesis` — state key mismatch                         | Refine step must also update `synthesis` to feed back into evaluate on next loop iteration     |

---

## Phase 2: Infrastructure Layer

**What**: Wrap Phase 1's DeepSummarizer with every production system: cost tracking, budget guards, lifecycle executor hooks, Instructor observability, retry logic, fallback chains, and caching. This phase makes agents production-safe.

**Depends on**: Phase 0 fully done. Phase 1's DeepSummarizer as a real agent to test against.

**Estimated scope**: Large (3–5 days)

**Status**: ✅ **COMPLETE** (Verified Mar 22, 2026)

---

### Tasks (in order)

#### T2.1 — `dopeagents/cost/tracker.py` — `CostTracker`

- File: `dopeagents/cost/tracker.py`
- Implement `CostTracker` (DD §8.1):
  - Thread-safe with `threading.Lock`
  - `record(agent, context, metrics)`: accumulates `metrics.cost_usd`
  - `get_agent_cost(agent_name)`, `get_total_cost()`, `get_summary()`
  - `CostTracker.noop()`: returns a no-op instance (for tests)

#### T2.2 — `dopeagents/cost/guard.py` — `BudgetGuard` + `BudgetConfig`

- File: `dopeagents/cost/guard.py`
- Implement `BudgetConfig` (DD §8.2):
  - `max_cost_per_call`, `max_cost_per_step`, `max_cost_per_agent`, `max_cost_global`
  - `max_refinement_loops`, `on_exceeded: Literal["error", "warn", "degrade"]`
- Implement `BudgetGuard`:
  - `check_pre_execution(agent, context, cost_tracker, budget)`: raises `BudgetExceededError` when `on_exceeded="error"`, raises `BudgetDegradedError` when `on_exceeded="degrade"`, logs warning when `on_exceeded="warn"`
  - `check_step_budget(step_name, step_cost, budget)`: per-step enforcement

> **Refinement note**: `BudgetDegradedError` (from T0.3) is raised when `on_exceeded="degrade"` — the guard returns the best-so-far result rather than raising a hard stop. This error must carry the degraded output so the caller can use it.

#### T2.3 — `dopeagents/observability/tracer.py` — tracer abstraction

- File: `dopeagents/observability/tracer.py`
- Implement the full tracer system (DD §7.1):
  - `Span` dataclass with `set_attribute()` and `add_event()`
  - `Tracer` ABC with `@contextmanager span()` method
  - `NoopTracer`: yields a `Span`, discards all data
  - `ConsoleTracer`: prints span open/close and final attributes to stdout
  - `Tracer.noop()` class method

#### T2.4 — `dopeagents/observability/otel.py` — OpenTelemetry tracer

- File: `dopeagents/observability/otel.py`
- Implement `OTelTracer` (DD §7.1) bridging `Span` to `opentelemetry-sdk` spans
- Guard with `try/except ImportError` → helpful error with `pip install dopeagents[otel]`

#### T2.5 — `dopeagents/observability/instructor_hooks.py` — Instructor hook callbacks

- File: `dopeagents/observability/instructor_hooks.py` (**new file — must be created**)
- Implement `InstructorObservabilityHooks` (DD §7.2):
  - `on_completion_kwargs(kwargs)`: set `llm.model` and `messages_count` on span
  - `on_completion_response(response)`: read `usage.prompt_tokens`, `usage.completion_tokens`, `_hidden_params.response_cost`; set on span
  - `on_completion_error(error)`: add span event
  - `on_parse_error(error)`: add span event for Instructor validation failures
  - `attach(client)`: wires all four callbacks via `client.on(...)`

> **Refinement note**: This file does not currently exist in the project. It must be created as a new file, not edited from a stub.

#### T2.6 — `dopeagents/lifecycle/hooks.py` — `LifecycleHooks`

- File: `dopeagents/lifecycle/hooks.py`
- Implement `LifecycleHooks` (DD §5.3) with no-op default implementations:
  - `pre_execution(agent, input, context)`
  - `post_execution(agent, input, output, context)`
  - `on_error(agent, input, error, context)`
  - `on_retry(agent, input, attempt, error, context)`
  - `on_fallback(original_agent, fallback_agent, context)`
  - `on_extraction_request(agent, messages, response_model, context)`
  - `on_extraction_response(agent, response, usage, context)`
  - `on_extraction_validation_error(agent, error, attempt, context)`

#### T2.7 — `dopeagents/lifecycle/executor.py` — `AgentExecutor`

- File: `dopeagents/lifecycle/executor.py`
- **Part A — Basic executor** (minimal `run()` that calls `agent.run()` with timing and error wrapping):
  - Constructor: accepts `tracer`, `cost_tracker`, `hooks`
  - `run(agent, input, context)`: time the call, wrap errors in `AgentExecutionError`, return `AgentResult`
- **Part B — Full lifecycle executor** (after T2.8–T2.12 are complete):
  - Extend constructor: add `cache_manager`, `budget`
  - Extend `run()` with full lifecycle:
    - **Pre**: input validation → `BudgetGuard.check_pre_execution()` → cache lookup → attach Instructor hooks
    - **Execute**: `_execute_with_retry()` wrapping `agent.run()`; if all retries fail and `fallback_chain` is set, call `_execute_fallback()`; else raise `AgentExecutionError`
    - **Post**: output validation → `_build_metrics_from_hooks()` → `cost_tracker.record()` → cache store → span attributes
  - `_attach_instructor_hooks(agent, span)`: wires `InstructorObservabilityHooks` to the agent's Instructor client
  - `_build_metrics_from_hooks(start_time, retry_count, fallback_used, span)`: collects tokens, cost, latency from span attributes

> **Refinement note**: Split into Part A (basic) and Part B (full lifecycle) to avoid a monolithic task. Part A can be done right after T2.1; Part B depends on T2.5, T2.8–T2.12 being ready. Both parts are in the same file.

#### T2.8 — `dopeagents/resilience/retry.py` — `RetryPolicy`

- File: `dopeagents/resilience/retry.py`
- Implement `RetryPolicy` (DD §9.1):
  - `max_attempts: int = Field(3, ge=1, le=10)`
  - `delay_seconds: float = 1.0`
  - `backoff_factor: float = 2.0`
  - `retryable_errors: list[type[Exception]] = [TimeoutError, ConnectionError]`

#### T2.9 — `dopeagents/resilience/fallback.py` — `FallbackChain`

- File: `dopeagents/resilience/fallback.py`
- Implement `FallbackChain` (DD §9.2):
  - Constructor validates output field compatibility between primary and fallback agents
  - `agents: list[Agent]` (ordered from most-capable to most-reliable)
  - Warn (not error) if fallback agent is missing fields the primary has

#### T2.10 — `dopeagents/resilience/degradation.py` — `DegradationChain`

- File: `dopeagents/resilience/degradation.py`
- Implement `DegradationResult` and `DegradationChain` (DD §9.2):
  - `DegradationChain` extends `FallbackChain`
  - `DegradationResult` fields: `output`, `agent_used: str`, `degradation_reason: str | None`
  - `run_with_degradation(input, context)` → `DegradationResult`
  - Warns if last agent in chain still requires LLM

#### T2.11 — `dopeagents/cache/manager.py` — `CacheManager` + `InMemoryCache`

- File: `dopeagents/cache/manager.py`
- Implement `CacheManager` ABC (DD §10):
  - Abstract: `get(agent, input)`, `set(agent, input, output, ttl)`, `invalidate(agent, input)`
  - `_build_key(agent, input)`: SHA-256 of `{agent.name, agent.version, input.model_dump()}`
- Implement `InMemoryCache`:
  - `dict[str, tuple[BaseModel, float | None]]` store
  - TTL enforcement in `get()`

#### T2.12 — `dopeagents/cache/disk.py` — `DiskCache`

- File: `dopeagents/cache/disk.py`
- Implement `DiskCache` backed by `diskcache` (DD §10)
- Guard with `try/except ImportError` → `pip install dopeagents[cache]`

#### T2.13 — `dopeagents/security/redaction.py` — `PIIRedactor`

- File: `dopeagents/security/redaction.py`
- Implement `PIIRedactor` (DD §21):
  - `PATTERNS` dict with regex for email, phone, SSN, credit card
  - `redact_fields(data, fields)`: nested dot-path field redaction
  - `redact_patterns(text)`: inline PII replacement in free text
- Wire into `AgentExecutor.run()` post-execution when `config.redact_pii_in_logs = True`

#### T2.14 — Infrastructure tests

- Files: `tests/cost/`, `tests/lifecycle/`, `tests/resilience/`, `tests/cache/`
- `CostTracker`: verify thread-safety under concurrent `record()` calls
- `BudgetGuard`: verify each `on_exceeded` mode raises the correct error type (including `BudgetDegradedError` for "degrade" mode)
- `AgentExecutor`:
  - Verify `InputValidationError` raised for invalid input
  - Verify `OutputValidationError` raised when `run()` returns wrong type
  - Verify retry count is captured in `ExecutionMetrics.retry_count`
  - Verify fallback agent is used when primary raises and `fallback_chain` is set
  - Verify cache hit skips `agent.run()` (mock to confirm zero calls)
- `PIIRedactor`: verify email/phone patterns are redacted in free text

#### T2.15 — Resilience integration test

- File: `tests/integration/test_resilience.py`
- End-to-end test: `AgentExecutor` with `RetryPolicy` + `FallbackChain` + `BudgetGuard` wired together against DeepSummarizer
- Verify that a budget-exceeded scenario with `on_exceeded="degrade"` returns a `BudgetDegradedError` carrying the best-so-far output
- Verify that retry exhaustion with a fallback chain correctly invokes the fallback agent

> **Refinement note**: This task was missing from the original roadmap. It validates that all resilience components integrate correctly as a system, not just individually.

---

### Definition of Done — Phase 2

- [X] `AgentExecutor().run(DeepSummarizer(), input)` returns `AgentResult` with non-zero `metrics.latency_ms`
- [X] `metrics.cost_usd` is populated from Instructor hooks (non-zero for real LLM calls)
- [X] `metrics.token_count_in` and `metrics.token_count_out` are captured from LiteLLM response
- [X] Budget exceeded correctly raises `BudgetExceededError` before any LLM call is made
- [X] Budget with `on_exceeded="degrade"` raises `BudgetDegradedError` (not `BudgetExceededError`)
- [X] `RetryPolicy(max_attempts=3)` retries exactly 3 times on `TimeoutError` then raises `AgentExecutionError`
- [X] `FallbackChain([DeepSummarizer(), fallback])` uses fallback when primary raises
- [X] `InMemoryCache` cache hit does not call `agent.run()` a second time
- [X] `PIIRedactor.redact_patterns("email me at user@example.com")` replaces the email
- [X] `mypy --strict dopeagents/lifecycle/ dopeagents/cost/ dopeagents/resilience/` exits 0
- [X] All infrastructure unit tests pass

**Status**: ✅ **COMPLETE** (Verified Mar 22, 2026)

**Implementation Notes**:
- Cost tracking fully thread-safe with `threading.Lock`
- Budget guards support three modes: `"error"`, `"warn"`, `"degrade"`
- Lifecycle executor wraps agents with lifecycle hooks, caching, and retry logic
- Instructor observability hooks capture tokens, cost, and latency per step
- Resilience layer: `RetryPolicy`, `FallbackChain`, `DegradationChain`
- Caching: `InMemoryCache` with TTL, `DiskCache` with `diskcache` backend
- Security: `PIIRedactor` with email/phone/SSN/credit card patterns
- All components thread-safe and production-ready

**Where Implemented**:
- [dopeagents/cost/](dopeagents/cost/) — Tracker, Guard, Budget configuration
- [dopeagents/lifecycle/](dopeagents/lifecycle/) — Executor, Hooks, Result models
- [dopeagents/resilience/](dopeagents/resilience/) — Retry, Fallback, Degradation chains
- [dopeagents/cache/](dopeagents/cache/) — Manager, InMemory, Disk implementations
- [dopeagents/observability/](dopeagents/observability/) — Tracer, Instructor hooks, Logging, OTel
- [dopeagents/security/](dopeagents/security/) — PII Redaction
- [tests/integration/test_resilience.py](tests/integration/test_resilience.py) — Full end-to-end resilience tests

---

### Risks / Unknowns — Phase 2

| Risk                                                                                                          | Mitigation                                                                                            |
| ------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Instructor's `client.on()` event API is undocumented; may not exist or may change                           | Wrap hook attachment in `try/except`; fall back to no-op; leave a test that asserts hook was called |
| `_hidden_params.response_cost` on LiteLLM response is provider-specific and may be `None`                 | Always coerce:`float(hidden.get("response_cost") or 0.0)`                                           |
| Thread-safety of `_get_client()` — two threads may race on lazy init                                       | Use `threading.Lock` (implemented in T0.7)                                                          |
| Multi-step agents run N `_extract()` calls per `run()` — metrics only capture last hook event by default | Sum all hook events into `ExecutionMetrics.steps` list; aggregate into top-level fields             |
| `observability/instructor_hooks.py` does not exist yet — it's a new file, not a stub                       | Create the file fresh in T2.5 (no stub to edit)                                                       |

---

## Phase 3: Second Agent — ResearchAgent + Composition

**What**: The second production agent (ResearchAgent, 6-step), plus the contract verification and pipeline composition system. Proves that two distinct multi-step agents compose correctly.

**Depends on**: Phase 0 (base classes), Phase 2 (infrastructure layer — lifecycle executor must exist to test pipeline end-to-end).

**Estimated scope**: Large (3–5 days)

**Status**: ✅ **COMPLETE**

---

### Tasks (in order)

#### T3.1 — `dopeagents/tools/base.py` — Tool ABC

- File: `dopeagents/tools/base.py`
- Implement `ToolInput`, `ToolOutput`, and abstract `Tool` (DD §16):
  - `call(input: ToolInput) → ToolOutput`
  - `input_type() → type[ToolInput]`
  - `to_llm_function_schema() → dict`

#### T3.2 — `dopeagents/tools/function.py` — `FunctionTool`

- File: `dopeagents/tools/function.py`
- Implement `FunctionTool` (DD §16): wraps a Python callable; introspects signature for JSON Schema
- Must implement `input_type()` (required by `Tool` ABC — return dynamically generated Pydantic model from function signature)
- Must implement `function_schema()` → dict (DD §16 reference implementation includes this method for generating JSON Schema from type hints via `_annotation_to_schema()`)

> **Refinement note**: The DD reference implementation includes a `function_schema()` method on `FunctionTool` that generates JSON Schema from function type hints. Additionally, `FunctionTool` must implement `input_type()` from the `Tool` ABC — the original roadmap omitted this. Generate a dynamic Pydantic model from the function's `inspect.signature()`.

#### T3.3 — `dopeagents/tools/rest.py` — `RESTTool`

- File: `dopeagents/tools/rest.py`
- Implement `RESTTool` (DD §16): `httpx`-backed REST call with `RESTToolConfig`
- Must implement `input_type()` from `Tool` ABC

#### T3.4 — Design the ResearchAgent LangGraph state

- File: `dopeagents/agents/research_agent.py`
- Define `ResearchState` TypedDict with all state keys shared across steps:
  - `query`, `max_sources`, `depth`
  - `sub_queries: list[str]` (formulate step out)
  - `raw_results: list[dict]` (search step out)
  - `scored_sources: list[dict]` (evaluate step out)
  - `draft_report: str` (synthesize step out)
  - `fact_check_notes: list[str]` (fact_check step out)
  - `final_report: str`, `citations: list[str]`, `confidence_scores: dict[str, float]`, `sub_queries_used: list[str]` (compose step out)
- Define all internal step output schemas (`_FormulateOut`, `_SearchOut`, `_EvaluateSourcesOut`, `_SynthesizeOut`, `_FactCheckOut`, `_ComposeOut`) as private Pydantic models.

#### T3.5 — Implement `ResearchInput` and `ResearchOutput`

- File: `dopeagents/agents/research_agent.py`
- `ResearchInput` (DD §17.2):
  - `query: str = Field(min_length=5)`
  - `max_sources: int = Field(default=10, ge=2, le=50)`
  - `depth: Literal["quick", "standard", "deep"] = "standard"`
- `ResearchOutput` (DD §17.2):
  - `report: str`, `sources: list[str]`, `confidence_scores: dict[str, float]`, `citations: list[str]`, `fact_check_notes: list[str]`, `sub_queries_used: list[str]`

#### T3.6 — Implement class-level metadata and step_prompts for ResearchAgent

- File: `dopeagents/agents/research_agent.py`
- Declare `step_prompts` for 6 steps (DD §17.2):
  - `formulate`: decompose the research question into 3–5 focused sub-queries (model strategy: Smart gpt-4o)
  - `search`: for each sub-query, produce candidate source titles, URLs, and relevance notes (model strategy: Fast/cheap + Tool)
  - `evaluate`: score each source for credibility (0–1) and relevance (0–1) (model strategy: Smart gpt-4o)
  - `synthesize`: write a draft report from the top-scored sources (model strategy: Smart gpt-4o)
  - `fact_check`: cross-reference 3–5 key claims against multiple sources; flag inconsistencies (model strategy: Smart gpt-4o)
  - `compose`: format the final report with inline citations and a confidence summary (model strategy: Fast/cheap gpt-4o-mini)

#### T3.7 — Implement all 6 step methods

- File: `dopeagents/agents/research_agent.py`
- Each step calls `self._extract(response_model=_<Name>Out, messages=[...], model=self._model_for_step("<name>"))`
- `_step_search`: simulates search by including sub-queries in the extraction prompt (real web search via `RESTTool` or `MCPTool` can be wired in later without changing the interface)
- Constructor accepts optional `tools: list[Tool]` parameter for wiring real search tools

#### T3.8 — Implement `_build_graph()` for ResearchAgent

- File: `dopeagents/agents/research_agent.py`
- Wire linear graph: `formulate → search → evaluate → synthesize → fact_check → compose → END`
- Override `_has_loops()` → `False`

#### T3.9 — Implement `run()` for ResearchAgent

- File: `dopeagents/agents/research_agent.py`
- Call `self._get_graph().invoke(initial_state)` and map final state to `ResearchOutput`

#### T3.10 — `dopeagents/contracts/types.py` — contract models

- File: `dopeagents/contracts/types.py`
- Implement `FieldMapping`, `CompatibilityResult`, `PipelineValidationError` (DD §4.3)

#### T3.11 — `dopeagents/contracts/checker.py` — `ContractChecker`

- File: `dopeagents/contracts/checker.py`
- Implement `ContractChecker.check(source, target, field_mappings)` (DD §4.4):
  - Field overlap check (Rule 1)
  - Required field coverage check (Rule 2)
  - Type compatibility check (Rule 3): `int → float` coercion allowed; `str → int` rejected
  - Returns `CompatibilityResult` with `compatible`, `mappings`, `errors`, `warnings`
  - Inner `TypeCompatibility` model
  - Document: union types are unsupported and deferred to a later pass

#### T3.12 — `dopeagents/contracts/pipeline.py` — `Pipeline`

- File: `dopeagents/contracts/pipeline.py`
- Implement `Pipeline` (DD §4.5):
  - Constructor validates sequential pairs at construction time
  - `field_mappings` dict for explicit overrides
  - `describe()`: multi-line string of step chain
  - `input_type` / `output_type` properties

#### T3.13 — Update `dopeagents/agents/__init__.py`

- File: `dopeagents/agents/__init__.py`
- Add exports: `ResearchAgent`, `ResearchInput`, `ResearchOutput`

#### T3.14 — Update `dopeagents/__init__.py`

- File: `dopeagents/__init__.py`
- Add: `DeepSummarizer`, `DeepSummarizerInput`, `DeepSummarizerOutput`, `ResearchAgent`, `ResearchInput`, `ResearchOutput`

> **Refinement note**: Also add the agent Input/Output types to root exports. `ContractChecker`, `Pipeline`, `SimpleRunner` should already be exported from T0.9 or Phase 2 work. Do not re-add items already present.

#### T3.15 — Contract and agent tests

- Files: `tests/contracts/`, `tests/agents/test_research_agent.py`
- `ContractChecker`: verify compatible pair passes, incompatible pair returns errors, optional field warning
- `Pipeline`: verify `Pipeline([A, B])` raises at construction (not at runtime) when types mismatch
- `ResearchAgent.describe().steps` == `["formulate", "search", "evaluate", "synthesize", "fact_check", "compose"]`
- `ResearchAgent.describe().has_loops` is `False`
- Integration: `Pipeline([DeepSummarizer, ResearchAgent])` raises `PipelineValidationError` (incompatible schemas — confirms the check is live)

---

### Definition of Done — Phase 3

- [X] `ResearchAgent().run(ResearchInput(query="..."))` returns a valid `ResearchOutput` with a real LLM
- [X] `ContractChecker.check(DeepSummarizer, ResearchAgent)` returns a `CompatibilityResult` with `compatible = False` and descriptive `errors`
- [X] `Pipeline([DeepSummarizer, ResearchAgent])` raises `ValueError` at construction
- [X] A `Pipeline` with two compatible agents (same output → input field names) validates and raises no errors
- [X] `FunctionTool` implements both `input_type()` and `function_schema()`
- [X] `mypy --strict dopeagents/agents/research_agent.py` exits 0
- [X] ResearchAgent init-time `step_prompts` customization working (identical to DeepSummarizer pattern)
- [X] **38 comprehensive tests (12 ContractChecker + 10 Pipeline + 16 Composition integration)**

**Status**: ✅ **COMPLETE** (Verified Mar 22, 2026)
- **Completion**: All contract verification (T3.10–T3.12) and composition integration tests (T3.15) passing

**Implementation Notes**:
- ResearchAgent implements 6-step workflow: `expand_query → search → analyze → synthesize → evaluate → refine`
- Same `step_prompts` customization pattern as DeepSummarizer (init-time, instance-level)
- Both agents consistently implement immutable `system_prompt` ClassVar + customizable `step_prompts`
- ResearchAgent.describe().has_loops = True (evaluate↔refine loop)
- Contract system (ContractChecker, Pipeline) validates agent composition at construction time
- Type compatibility supports exact match and allowed coercions (int→float, bool→int)
- **Where Implemented**:
  - [dopeagents/contracts/types.py](dopeagents/contracts/types.py) — Type and contract models
  - [dopeagents/contracts/checker.py](dopeagents/contracts/checker.py) — Field mapping and compatibility validation
  - [dopeagents/contracts/pipeline.py](dopeagents/contracts/pipeline.py) — Sequential agent composition
  - [tests/contracts/test_checker.py](tests/contracts/test_checker.py) — 12 ContractChecker tests
  - [tests/contracts/test_pipeline.py](tests/contracts/test_pipeline.py) — 10 Pipeline validation tests
  - [tests/integration/test_composition.py](tests/integration/test_composition.py) — 16 composition integration tests

---

### Risks / Unknowns — Phase 3

| Risk                                                                                                                                                   | Mitigation                                                                                                                                                       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ResearchAgent needs real web search to be valuable —`_step_search` with only LLM simulation produces low-quality results                            | Clearly document that web search is a stub; expose a `tools: list[Tool]` parameter on `__init__` for wiring real search tools without changing the interface |
| `ContractChecker` type compatibility is simplistic — Pydantic's type system (unions, generics, Annotated) is much richer than a dict of annotations | Start with exact match +`int → float` coercion; document that union types are unsupported and deferred to a later pass                                        |
| `FunctionTool` doesn't implement `input_type()` from `Tool` ABC                                                                                  | Must generate a dynamic Pydantic model from the function's `inspect.signature()`                                                                               |

---

## Phase 4: MCP Exposure

**What**: Expose any DopeAgents agent as an MCP tool, prompt, and resource via FastMCP. This makes agents discoverable and callable from Claude Desktop, Cursor, and any MCP-compliant client.

**Depends on**: Phase 0 (base class + types), Phase 1 (at least one real agent to test with).

**Estimated scope**: Medium (1–2 days)

---

### Tasks (in order)

#### T4.1 — `dopeagents/adapters/mcp.py` — full implementation

- File: `dopeagents/adapters/mcp.py`
- Implement all functions from DD §11.9:
  - `register_agent_as_mcp_tool(agent, mcp_server, description_override)`: async tool handler, validates input, calls `agent.run()`, returns `result.model_dump()`
  - `register_agent_prompt(agent, mcp_server)`: exposes `agent.debug(input).prompt` as MCP Prompt
  - `register_agent_catalog_resource(agents, mcp_server)`: read-only resource at `dopeagents://catalog`
  - `create_single_agent_mcp_server(agent, name)`: single-agent FastMCP server
  - `create_mcp_server(agents, name)`: multi-agent server; if `agents=None`, loads all built-ins via `SandboxRunner`
- Guard all functions with `_check_installed()` → `MCPNotInstalledError` with install hint: `pip install dopeagents[mcp]`

> **Refinement note**: The MCP catalog resource URI is `dopeagents://catalog` (not `agentmaker://catalog`). MCP tool name must be sanitized to `snake_case` from `agent.name`; raise `MCPRegistrationError` on name collision.

#### T4.2 — Wire `Agent.as_mcp_tool()` and `Agent.as_mcp_server()`

- File: `dopeagents/core/agent.py`
- Implement the two MCP adapter methods inherited by all agents:
  ```python
  def as_mcp_tool(self, mcp_server, **kwargs):
      from dopeagents.adapters.mcp import register_agent_as_mcp_tool
      return register_agent_as_mcp_tool(self, mcp_server, **kwargs)

  def as_mcp_server(self, name=None, **kwargs):
      from dopeagents.adapters.mcp import create_single_agent_mcp_server
      return create_single_agent_mcp_server(self, name=name, **kwargs)
  ```
- Both are already declared as stubs in T0.7; this task implements the bodies.

#### T4.3 — MCP tests (without a live server)

- File: `tests/adapters/test_mcp.py`
- Verify `register_agent_as_mcp_tool` registers a tool on a mock FastMCP server (inspect the tool registry, not the transport)
- Verify `MCPNotInstalledError` is raised when `fastmcp` is not importable (mock `importlib`)
- Verify `create_mcp_server(agents=[DeepSummarizer()])` returns an object with a `.run()` method
- Verify `register_agent_catalog_resource` returns correct metadata for each agent at `dopeagents://catalog` (no network calls)
- Verify MCP tool names are snake_case sanitized
- Verify `MCPRegistrationError` on tool name collision

#### T4.4 — MCP integration test: Claude Desktop schema validation

- File: `tests/integration/test_mcp_schema.py`
- Start an MCP server in-process with `transport="stdio"` (use FastMCP's test client if available)
- Call the `DeepSummarizer` tool with a valid input
- Assert the output matches `DeepSummarizerOutput` schema
- This test confirms Claude Desktop would receive a valid tool schema

---

### Definition of Done — Phase 4

- [ ] `DeepSummarizer().as_mcp_server().run(transport="stdio")` starts without error
- [ ] `DeepSummarizer().as_mcp_tool(mcp)` registers the tool on a FastMCP instance
- [ ] The MCP tool input schema exactly matches `DeepSummarizerInput.model_json_schema()`
- [ ] `register_agent_catalog_resource` returns metadata for all registered agents at `dopeagents://catalog`
- [ ] `MCPNotInstalledError` is raised with the correct install command (`pip install dopeagents[mcp]`) when FastMCP is absent
- [ ] A Claude Desktop `mcp.json` config pointing to `dopeagents mcp serve` can successfully discover and call `DeepSummarizer` (manual verification)

---

### Risks / Unknowns — Phase 4

| Risk                                                                                                    | Mitigation                                                                                                                |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| FastMCP 3.x API differences from prior versions —`@mcp_server.tool()` decorator signature may differ | Pin `fastmcp >= 3.0, < 4.0`; review FastMCP changelog before implementing                                               |
| Async tool handler (`async def agent_tool(**kwargs)`) — `agent.run()` is synchronous               | Wrap `agent.run()` in `asyncio.get_event_loop().run_in_executor(None, ...)` if FastMCP requires non-blocking handlers |
| MCP tool name conflicts when two agents share similar names                                             | Sanitize `agent.name` to `snake_case` tool name; raise `MCPRegistrationError` on collision                          |

---

## Phase 5: CLI, Sandbox & Polish

**What**: The developer experience layer. `dopeagents list/describe/run/dry-run`, interactive REPL, `wrap_function`/`wrap_class`, framework adapter implementations, packaging, and a final README.

**Depends on**: All prior phases.

**Estimated scope**: Medium (2–3 days)

---

### Tasks (in order)

#### T5.1 — `dopeagents/sandbox/runner.py` — `SandboxRunner`

- File: `dopeagents/sandbox/runner.py`
- Implement `SandboxRunner` (DD §6.3):
  - `_BUILTIN_AGENTS` dict lazily populated via `_register_builtins()`
  - `load(name_or_class)` → instantiated agent (accepts str, class, or instance)
  - `list_agents()` → list of metadata dicts
  - `describe(agent)` → `AgentMetadata.model_dump()`
  - `dry_run(agent, **kwargs)` → `DebugInfo`
  - `run(agent, context, **kwargs)` → `AgentResult` via `AgentExecutor`
  - `compare(agents, **kwargs)` → `ComparisonResult`
  - `inspect_mcp(agent)` → dict with `mcp_tool_name`, `mcp_tool_description`, `mcp_input_schema`, `mcp_output_schema`, `mcp_capabilities`

#### T5.2 — `dopeagents/sandbox/display.py` — `SandboxDisplay`

- File: `dopeagents/sandbox/display.py`
- Implement `SandboxDisplay` (DD §6.4) — pure text formatting, no side effects:
  - `format_describe(metadata)` → multi-line agent info + schemas
  - `format_dry_run(debug_info)` → prompt, schemas, extraction mode, "NO API CALL MADE"
  - `format_result(result)` → output fields + metrics (cost, latency, tokens, cache)
  - `format_comparison(comparison)` → tabular sorted by latency
  - `format_mcp_inspect(mcp_info)` → tool name, description, schemas

#### T5.3 — `dopeagents/sandbox/repl.py` — Interactive REPL

- File: `dopeagents/sandbox/repl.py`
- Implement `launch_repl()` (DD §6.6) using `code.InteractiveConsole`:
  - Pre-loaded functions: `load`, `describe`, `dry_run`, `run`, `compare`, `inspect_mcp`, `list_agents`, `help`
  - Banner shows agent count
  - No IPython dependency

#### T5.4 — `dopeagents/cli.py` — sandbox CLI commands

- File: `dopeagents/cli.py`
- Implement full CLI with `click`:
  - `mcp serve` command (existing from Phase 4)
  - Add `sandbox` command group with subcommands (DD §6.5):
    - `sandbox list` → `runner.list_agents()` + `SandboxDisplay`
    - `sandbox describe <agent_name>` → `runner.describe()` + `SandboxDisplay.format_describe()`
    - `sandbox dry-run <agent_name> --input <json>` → `runner.dry_run()` + `format_dry_run()`
    - `sandbox run <agent_name> --input <json> [--model <model>] [--output-format text|json]`
    - `sandbox compare <a1> <a2> ... --input <json>`
    - `sandbox inspect-mcp <agent_name>`
    - `sandbox repl` → `launch_repl()`

#### T5.5 — `dopeagents/adapters/wrap.py` — `wrap_function` / `wrap_class`

- File: `dopeagents/adapters/wrap.py`
- Implement `wrap_function()` (DD §12.1):
  - Accepts `func, name, version, input_type, output_type, description, capabilities, requires_llm`
  - Returns a new concrete `Agent` subclass that delegates to `func`
  - Output coercion: if func returns `dict`, validate to `output_type`; if returns `str`, put in first field
- Implement `wrap_class()`: instantiates `cls`, binds the method, delegates to `wrap_function()`

#### T5.6 — `dopeagents/adapters/langgraph.py` — LangGraph adapter

- File: `dopeagents/adapters/langgraph.py`
- Implement `to_langgraph_node()` (DD §11.2):
  - Accepts `agent`, `input_mapping: dict[str, str] | None`, `output_mapping: dict[str, str] | None`, `context_factory: callable | None`
  - Returns a node function `(state: dict) → dict` suitable for LangGraph `StateGraph`
  - Input mapping: `{agent_field: state_key}` — maps agent input fields to state keys
  - Output mapping: `{state_key: agent_field}` — maps state keys to agent output fields
  - Sets `node_function.__name__` to `dopeagents_{agent.name}`
  - Attaches `_dopeagents_agent` attribute for introspection
- Guard with `_check_installed()` → `FrameworkNotInstalledError` with `pip install dopeagents[langgraph]`

> **Refinement note**: This task was missing from the original roadmap. DD §11.2 provides a complete reference implementation for converting DopeAgents agents into LangGraph node functions with state key mapping. This is critical for users who want to embed DopeAgents agents within their own LangGraph workflows.

#### T5.7 — `dopeagents/adapters/langchain.py` — LangChain adapters

- File: `dopeagents/adapters/langchain.py`
- Implement `to_langchain_runnable()` and `to_langchain_tool()` (DD §11.3)
- Guard with `_check_installed()` → `FrameworkNotInstalledError`

#### T5.8 — `dopeagents/adapters/crewai.py` — CrewAI adapter

- File: `dopeagents/adapters/crewai.py`
- Implement `to_crewai_tool()` (DD §11.4)
- Guard with `_check_installed()`

#### T5.9 — `dopeagents/adapters/autogen.py` — AutoGen adapter

- File: `dopeagents/adapters/autogen.py`
- Implement `to_autogen_function()` (DD §11.5) returning `{function, name, description, parameters}`
- Guard with `_check_installed()`

#### T5.10 — `dopeagents/adapters/openai_functions.py` — OpenAI function schema

- File: `dopeagents/adapters/openai_functions.py`
- Implement `to_openai_function()` (DD §11.6): returns `{schema, callable}`

#### T5.11 — `dopeagents/adapters/generic.py` — plain callable adapter

- File: `dopeagents/adapters/generic.py`
- Implement `to_callable(agent, input_format, output_format)` (DD §11.7)

#### T5.12 — `dopeagents/adapters/simple.py` — `SimpleRunner`

- File: `dopeagents/adapters/simple.py`
- Implement `SimpleRunner` (DD §11.8): minimal convenience wrapper around `AgentExecutor`

#### T5.13 — `dopeagents/spec/schema.py` — `AgentSpec`

- File: `dopeagents/spec/schema.py`
- Implement `AgentSpec` Pydantic model (DD §13.2)

#### T5.14 — `dopeagents/spec/generator.py` — `SpecGenerator`

- File: `dopeagents/spec/generator.py`
- Implement `SpecGenerator.generate(agent_class)` → `AgentSpec` (DD §13.3)
- `to_yaml(spec)` — guarded by `try/except ImportError` for PyYAML
- `to_json(spec)` — uses `spec.model_dump_json()`

#### T5.15 — `dopeagents/spec/validator.py` — `SpecValidator`

- File: `dopeagents/spec/validator.py`
- Implement `SpecValidator.validate(agent_class)` → `list[str]` errors (DD §13.4):
  - Check required class attrs, `input_type()` / `output_type()` resolve, `run()` signature, semver format

#### T5.16 — `dopeagents/registry/registry.py` — `Registry` + `@register`

- File: `dopeagents/registry/registry.py`
- Implement `Registry` class (DD §14) with class-level state:
  - `register(agent_class)`: validates via `SpecValidator`, generates `AgentSpec`, indexes by capability and tag
  - `find(capability)`, `search(tags, requires_llm)`, `get(name, version)`, `list_all()`, `clear()`
  - `register` decorator
- Uses `packaging.version.Version` for semver parsing — `packaging` must be in core dependencies (see T0.1)

> **Refinement note**: `Registry.get()` uses `packaging.version.Version` for version parsing. The `packaging` library must be listed in core dependencies (T0.1) — not just as a transitive dependency.

#### T5.17 — `dopeagents/benchmark/suite.py` and `dopeagents/benchmark/results.py`

- Files: `dopeagents/benchmark/suite.py`, `dopeagents/benchmark/results.py`
- Implement `BenchmarkCase`, `BenchmarkSuite` (DD §15), and `AgentBenchmarkResult`, `BenchmarkResult` (DD §15)

#### T5.18 — `dopeagents/benchmark/runner.py` — `BenchmarkRunner`

- File: `dopeagents/benchmark/runner.py`
- Implement `BenchmarkRunner.run_single()` and `compare()` (DD §15)
- Calls `agent.run()` directly — intentionally bypasses `AgentExecutor` to measure raw agent performance

#### T5.19 — Final `dopeagents/__init__.py` re-exports

- File: `dopeagents/__init__.py`
- Add all remaining exports per DD §18.1:
  ```python
  from dopeagents.agents.deep_summarizer import DeepSummarizer, DeepSummarizerInput, DeepSummarizerOutput
  from dopeagents.agents.research_agent import ResearchAgent, ResearchInput, ResearchOutput
  from dopeagents.contracts.checker import ContractChecker
  from dopeagents.contracts.pipeline import Pipeline
  from dopeagents.adapters.simple import SimpleRunner
  from dopeagents.registry.registry import Registry, register
  ```

#### T5.20 — `README.md` — accurate and complete

- File: `README.md`
- Must reflect what is actually implemented post-Phase 5
- Sections: Installation, Quick Start (DeepSummarizer, ResearchAgent), MCP exposure, Sandbox CLI, Framework adapters, Configuration, Contributing
- All import paths use `dopeagents.*` (not `agentmaker.*`)

#### T5.21 — CLI & sandbox tests

- Files: `tests/test_cli.py`, `tests/sandbox/`
- `dopeagents sandbox list` outputs at least one agent name
- `dopeagents sandbox describe DeepSummarizer` outputs input/output schema fields
- `dopeagents sandbox dry-run DeepSummarizer --input '{"text":"..."}'` outputs "NO API CALL MADE"
- `wrap_function()` result passes `ContractChecker.check()` and `SpecValidator.validate()`
- `Registry.register()` followed by `Registry.get("DeepSummarizer")` returns the correct class
- Adapter tests marked with `pytest.mark.skipif(not importlib.util.find_spec("langchain_core"), reason="langchain not installed")`

#### T5.22 — `pip install dopeagents` smoke test

- File: `tests/test_smoke.py`
- `from dopeagents import Agent, DeepSummarizer, ResearchAgent, ContractChecker, Pipeline, SimpleRunner, Registry, register`
- `DeepSummarizer().describe()` works without an API key
- `AgentExecutor` is importable from `dopeagents.lifecycle`
- CLI entry point: `dopeagents --help` exits 0

---

### Definition of Done — Phase 5

- [ ] `pip install .` succeeds cleanly
- [ ] `dopeagents --help` and `dopeagents sandbox --help` display correctly
- [ ] `dopeagents sandbox list` lists DeepSummarizer and ResearchAgent
- [ ] `dopeagents sandbox dry-run DeepSummarizer --input '{"text": "hello"}'` outputs prompt preview with "NO API CALL MADE"
- [ ] `dopeagents mcp serve --transport stdio` starts and prints the MCP schema handshake
- [ ] `wrap_function()` produces an Agent that passes `SpecValidator.validate()`
- [ ] `Registry.register(DeepSummarizer); Registry.find("summarization")` returns DeepSummarizer's spec
- [ ] `BenchmarkRunner().run_single(DeepSummarizer(), suite)` completes without error using mocked `_extract()`
- [ ] `to_langgraph_node()` converts an agent to a valid LangGraph node function
- [ ] `README.md` quick-start code block works without modification (uses `dopeagents.*` imports)
- [ ] `mypy --strict dopeagents/` exits 0 (or with a documented, intentional ignore list)
- [ ] `pytest tests/ --ignore=tests/integration` passes (all non-LLM tests green)

---

### Risks / Unknowns — Phase 5

| Risk                                                                                                           | Mitigation                                                                                                   |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| LangChain, CrewAI, AutoGen adapter tests require those packages installed — CI matrix complexity              | Mark adapter tests with `pytest.mark.skipif(not importlib.util.find_spec("langchain_core"), ...)`          |
| `code.InteractiveConsole` REPL behavior differs on Windows vs Unix for `input()` and `KeyboardInterrupt` | Test REPL `launch_repl()` is callable and returns without error; leave interactive behavior as manual test |
| Registry uses class-level mutable state — parallel test runs interfere                                        | Call `Registry.clear()` in a `pytest` fixture's teardown                                                 |
| README accuracy — docs rot quickly                                                                            | Add a CI check that runs the README quick-start code block against a mock LLM                                |
| `packaging` library used by Registry but not a standard dependency                                           | Added to core dependencies in T0.1                                                                           |

---

## Cross-Phase Invariants

These constraints apply across all phases and must never be broken by any commit:

| ID     | Invariant                                                                                                           | Violated by                                                                   |
| ------ | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| INV-1  | Every Agent subclass has explicit Pydantic `InputT` and `OutputT`. No raw dicts cross agent boundaries.         | Shortcuts in step methods; bare `dict` returns from `run()`               |
| INV-2  | `Agent.run()` never mutates `self` or external state. Model overrides are passed to `_extract()`, not stored. | Setting `self._model = context.model_override` inside `run()`             |
| INV-3  | Every LLM call goes through `self._extract()`. No raw `openai` / `anthropic` calls inside agents.             | Direct SDK calls inside step methods                                          |
| INV-4  | Every agent failure raises a typed error from `errors.py`. No bare `raise Exception(...)`.                      | Any `raise Exception(...)` or `raise ValueError(...)` inside agent code   |
| INV-5  | Observability, cost, and retry live in the lifecycle layer. Not inside `run()`.                                   | Adding `try/except` retry loops inside `run()` or step methods            |
| INV-6  | Framework adapters never modify agent behavior. Only translate I/O.                                                 | Logic inside adapter `invoke()` or `_run()` methods beyond field mapping  |
| INV-7  | LangChain, CrewAI, AutoGen, and LangGraph adapters are always optional and lazy-imported.                           | `import langchain` at module top level in any adapter                       |
| INV-8  | The internal LangGraph graph is never exposed publicly. Callers use `run()`.                                      | Exposing `self._graph` or `_build_graph()` via public attribute or method |
| INV-9  | MCP adapters do not modify agent behavior. The same `agent.run()` executes regardless of caller.                  | Adding MCP-specific logic inside `agent.run()`                              |
| INV-10 | Instructor is an internal implementation detail. Neither agent authors nor users import it directly.                | `import instructor` in any agent subclass or user-facing API                |

---

## Test Strategy

| Level                  | How                                                                      | When                       |
| ---------------------- | ------------------------------------------------------------------------ | -------------------------- |
| Unit (no LLM)          | Mock `_extract()` to return canned Pydantic models                     | Every PR; CI on every push |
| Contract               | `ContractChecker` validation; `Pipeline` construction-time errors    | Phase 3+                   |
| Integration (live LLM) | `pytest -m integration` requires `OPENAI_API_KEY`                    | Manual or nightly CI       |
| Smoke                  | Import all public symbols;`describe()` and `debug()` without API key | Every PR                   |
| CLI                    | `click.testing.CliRunner` invocations                                  | Phase 5+                   |

```
# Run only unit tests (no API key needed):
pytest tests/ --ignore=tests/integration -v

# Run integration tests (requires API key):
pytest tests/integration -v -m integration

# Type check:
mypy --strict dopeagents/

# Full check:
pytest && mypy --strict dopeagents/
```
