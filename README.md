# DopeAgents

> Production-grade AI agents — typed interfaces, hybrid LLM pipelines, zero mandatory API keys.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/dopeagents.svg)](https://pypi.org/project/dopeagents/)

---

## What is it?

DopeAgents is a Python framework for building **typed, multi-step AI agents** on top of LangGraph and LiteLLM. Each agent is a deterministic pipeline — some steps run real code, others call an LLM — so behaviour is transparent, testable, and cost-controlled.

**Two production agents are included out of the box:**

| Agent | Steps | What it does |
|---|---|---|
| `DeepSummarizer` | 7 | Chunk → summarize → synthesize → self-evaluate → refine |
| `DeepResearcher` | 13 | Search 7 free APIs → extract → score credibility → analyse → cite → report |

`DeepResearcher` requires **zero API keys** for its search layer (Wikipedia, DuckDuckGo, arXiv, CrossRef, OpenLibrary, US Gov, Library of Congress).

---

## Install

```bash
pip install dopeagents          # core
pip install dopeagents[research] # + web content extraction
pip install dopeagents[all]      # everything
```

Requires **Python 3.12+**.

---

## Quick start

```python
from dopeagents import DeepSummarizer
from dopeagents.agents import DeepSummarizerInput

summarizer = DeepSummarizer()

result = summarizer.run(
    DeepSummarizerInput(
        text="Paste your document here...",
        max_length=300,
        style="bullets",       # paragraph | bullets | tldr
        focus="key findings",
        quality_threshold=0.8,
    )
)

if result.success:
    print(result.output.summary)
    print(f"Quality: {result.output.quality_score:.2f}")
    print(f"Cost:    ${result.metrics.total_cost:.4f}")
```

---

## Configuration

All settings come from `DOPEAGENTS_*` environment variables (or a `.env` file).

```bash
DOPEAGENTS_DEFAULT_MODEL=gpt-4o          # any LiteLLM model string
DOPEAGENTS_API_KEY=sk-...                # forwarded to LiteLLM
DOPEAGENTS_MAX_COST_PER_CALL=5.00        # hard budget per run()
DOPEAGENTS_MAX_RETRIES=3                 # auto-retry with backoff
DOPEAGENTS_LOG_LEVEL=INFO                # DEBUG|INFO|WARNING|ERROR
DOPEAGENTS_LOG_COLOR=true                # false in production/CI
DOPEAGENTS_ENABLE_CACHE=false            # memory|disk (set backend below)
DOPEAGENTS_CACHE_BACKEND=memory
```

The framework auto-detects provider keys (`OPENAI_API_KEY`, `GROQ_API_KEY`, `NVIDIA_NIM_API_KEY`, …) — no extra wiring needed.

---

## Agents in depth

### DeepSummarizer

Seven steps, four LLM calls, three deterministic code steps:

1. Analyse text structure → derive chunk size (code)
2. Split into cost-bounded chunks (code)
3. Summarise each chunk (LLM)
4. Synthesise chunk summaries (LLM)
5. Score: faithfulness · completeness · coherence (LLM)
6. Refine using targeted feedback (LLM, up to `max_refinement_loops`)
7. Apply style and truncate to `max_length` (code)

### DeepResearcher

Thirteen steps, hybrid code + bounded LLM tool calling:

1. Load prior session context (code)
2. Expand query and plan strategy (LLM)
3. Search 7 free APIs in parallel (code)
4. Extract full content from top URLs (code)
5. Score source credibility: domain authority, citations, recency (code)
6. Extract claims with bounded tool calls — fact-check, cite, search (LLM + tools)
7. Cross-reference claims: agreement / contradiction (LLM)
8. Synthesise evidence with inline citations (LLM)
9. Calculate grounded confidence score (code)
10. Evaluate report quality, detect gaps (LLM)
11. Gap-fill with targeted searches (LLM + code)
12. Generate structured report — markdown, JSON, or plain text (LLM + code)
13. Persist session for follow-up queries (code)

> **Status:** fully implemented and tested; not yet exported from the top-level `dopeagents` namespace. Import from `dopeagents.agents.deep_researcher`.

---

## Project layout

```
dopeagents/
├── core/           # Agent base class, types, context
├── agents/         # DeepSummarizer, DeepResearcher + internals
├── agent_utils/    # Search providers, content extraction, credibility
├── observability/  # Logging, tracing, step metrics
├── cost/           # Budget guards and cost tracker
├── cache/          # Memory and disk backends
├── resilience/     # Retry with exponential backoff
├── adapters/       # LangChain, CrewAI, AutoGen adapters
├── tools/          # Tool definitions and execution
└── config.py       # DopeAgentsConfig (env-based)
```

---

## Development

```bash
git clone https://github.com/yourusername/DopeAgents.git
cd DopeAgents

# install with all extras + dev tools
uv sync --all-extras          # or: pip install -e ".[dev]"

# pre-commit hooks (mypy, ruff, black, isort)
pre-commit install

# tests
pytest --cov=dopeagents

# type check + lint
mypy dopeagents
ruff check dopeagents
```

---

## Optional extras

| Extra | Installs |
|---|---|
| `research` | `trafilatura`, `beautifulsoup4`, `duckduckgo-search` |
| `cache` | `diskcache` |
| `otel` | OpenTelemetry SDK + OTLP exporter |
| `mcp` | `fastmcp` (MCP server) |
| `langchain` | LangChain adapter |
| `crewai` | CrewAI adapter |
| `autogen` | AutoGen adapter |
| `all` | Everything above |

---

## Contributing

1. Fork → feature branch → PR
2. Tests must pass: `pytest`
3. Types must pass: `mypy dopeagents`
4. Follow existing code style (ruff + black enforced by pre-commit)

See [docs/](docs/) for architecture and design documents.

---

## License

MIT — see [LICENSE](LICENSE).

---

Built with [LangGraph](https://github.com/langchain-ai/langgraph) · [LiteLLM](https://github.com/BerriAI/litellm) · [Pydantic](https://docs.pydantic.dev/) · [Instructor](https://github.com/jxnl/instructor) · [Rich](https://rich.readthedocs.io/)
