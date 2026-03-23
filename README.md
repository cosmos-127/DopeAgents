# DopeAgents

> Production-grade AI agents with typed interfaces and sophisticated multi-step workflows

**DopeAgents** is a framework for building robust, type-safe AI agents that scale. It provides a solid foundation for orchestrating complex multi-step workflows with LLM models while maintaining code quality and production readiness.

## Features

- **Typed Interfaces**: Full type hints and Pydantic-based validation for type-safe agent interactions
- **Multi-step Workflows**: Sophisticated orchestration using LangGraph for complex agent behaviors
  - DeepSummarizer: 7-step document summarization with iterative refinement
  - DeepResearcher: 13-step research with real search APIs and bounded tool calling
- **Real Search Integration**: Actual free API search (Wikipedia, DuckDuckGo, Semantic Scholar, arXiv, CrossRef) instead of hallucinations
- **Hybrid Pipeline Architecture**: Combination of deterministic code steps with adaptive LLM-based analysis and bounded tool calling
- **Cost Tracking**: Per-step and global cost monitoring with budget enforcement (error or graceful degradation)
- **Observability**: Structured logging, tracing, step-level metrics collection, and debug information
- **Error Handling**: Comprehensive, typed error hierarchy with JSON serialization for structured error handling
- **Configuration Management**: Environment-based configuration with auto-detection of LLM providers and sensible defaults
- **Colored Logging**: Rich, environment-aware logging (ANSI colors in development, plain text in production)
- **Extensible Architecture**: Modular design for custom adapters, tools, and caching backends
- **Research Memory**: Session persistence and context loading for DeepResearcher follow-up queries
- **Credibility Scoring**: Evidence-based confidence calculation from measurable signals (domain authority, citations, recency)

## Built-in Agents

DopeAgents includes production-ready multi-step agents:

### DeepSummarizer
A 7-step summarization agent with self-evaluation and iterative refinement:
- Text structure analysis and chunk size heuristic (code)
- Document chunking into cost-bounded chunks (code)
- Chunk summarization (LLM)
- Synthesis of chunk summaries (LLM)
- Quality evaluation with faithfulness, completeness, coherence scoring (LLM)
- Iterative refinement loop (LLM)
- Output formatting and style application (code)

Features configurable summary styles (paragraph, bullets, tldr) and optional focus areas.

### DeepResearcher
A 13-step hybrid research agent with code-controlled pipeline and bounded LLM tool calling:
- Context loading from previous research sessions (code)
- Query expansion and strategy determination (LLM)
- Real search across Wikipedia, DuckDuckGo, Semantic Scholar, arXiv, CrossRef (code)
- Content extraction from source URLs (code)
- Credibility scoring via domain authority, citations, recency (code)
- Deep analysis with bounded tool calling for fact-checking and citations (LLM + tools)
- Cross-referencing claims for agreement/contradiction (LLM)
- Evidence-based synthesis with citations (LLM)
- Grounded confidence calculation from measurable signals (code)
- Quality evaluation and gap detection (LLM)
- Targeted gap-filling refinement (LLM + code)
- Structured report generation in multiple formats (LLM + code)
- Session persistence for follow-up queries (code)

Note: DeepResearcher is implemented but not yet exported from the top-level package API.

Both agents are fully typed, extensively tested, and ready for production use.

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install dopeagents

# Or with pip
pip install dopeagents
```

**Requires Python 3.12+**

### Basic Usage

```python
from dopeagents import Agent, get_logger, get_config

# Get a colored logger
logger = get_logger(__name__)

# Access configuration (reads from DOPEAGENTS_* environment variables)
config = get_config()

# Build and run your agent
agent = Agent(name="example", description="My first agent")
result = agent.run(input_data={"task": "extract insights"})

logger.info(f"Agent completed: {result.success}")
```

## Configuration

DopeAgents uses environment variables for configuration. All settings are read from `DOPEAGENTS_*` prefixed environment variables (case-insensitive).

### Common Configuration

```bash
# Model and API
DOPEAGENTS_DEFAULT_MODEL=gpt-4o
DOPEAGENTS_API_KEY=your-key-here

# Logging
DOPEAGENTS_LOG_COLOR=true           # Enable colored output (default: true)
DOPEAGENTS_LOG_LEVEL=INFO           # Log level: DEBUG|INFO|WARNING|ERROR|CRITICAL

# Cost Management
DOPEAGENTS_ENABLE_COST_TRACKING=true
DOPEAGENTS_MAX_COST_PER_CALL=10.00  # Max USD per agent.run() call

# Observability
DOPEAGENTS_TRACER_TYPE=console      # console|otel|noop
DOPEAGENTS_ENABLE_STEP_METRICS=true

# Caching
DOPEAGENTS_ENABLE_CACHE=false
DOPEAGENTS_CACHE_BACKEND=memory     # memory|disk

# Resilience
DOPEAGENTS_ENABLE_RETRY=true
DOPEAGENTS_MAX_RETRIES=3
DOPEAGENTS_RETRY_BASE_DELAY_SECONDS=1.0
```

### Logging Configuration

The logging system is environment-aware:

- **Development** (`DOPEAGENTS_LOG_COLOR=true`): Rich, colorful console output
- **Production** (`DOPEAGENTS_LOG_COLOR=false`): Plain text format suitable for log files and structured logging

```python
from dopeagents import get_logger

# Get a configured logger
logger = get_logger(__name__)

logger.debug("Detailed diagnostic info")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### Project Structure

```
dopeagents/
├── core/                  # Agent base class and core types
├── agents/                # Concrete agent implementations
│   ├── deep_summarizer.py # 7-step summarization agent
│   ├── deep_researcher.py # 13-step research agent
│   ├── _summarizer/       # DeepSummarizer internals
│   └── _researcher/       # DeepResearcher internals
├── agent_utils/           # Shared utilities (search, extraction, credibility, etc.)
├── config.py              # Configuration management (environment-based)
├── observability/         # Logging, tracing, metrics, instrumentation
├── adapters/              # LLM provider integrations (LiteLLM, LangGraph)
├── tools/                 # Tool definitions and execution
├── cache/                 # Caching backends (memory, disk)
├── cost/                  # Cost tracking and budget management
├── errors.py              # Structured error hierarchy
└── __init__.py            # Public API exports
```

## Key Components

### Agent Execution Example

```python
from dopeagents import DeepSummarizer, get_logger, AgentResult, ExecutionMetrics
from dopeagents.agents import DeepSummarizerInput

logger = get_logger(__name__)

# Initialize the summarizer
summarizer = DeepSummarizer()

# Run the 7-step summarization workflow
result: AgentResult[DeepSummarizerOutput] = summarizer.run(
    DeepSummarizerInput(
        text="Your long document text here...",
        max_length=500,
        style="bullets",
        focus="key insights",
        quality_threshold=0.8,
        max_refinement_loops=3
    )
)

if result.success:
    output = result.output
    logger.info(f"Summary: {output.summary}")
    logger.info(f"Key Points: {output.key_points}")
    logger.info(f"Quality Score: {output.quality_score}")
    logger.info(f"Refinement Rounds: {output.refinement_rounds}")
    logger.info(f"Chunks Processed: {output.chunks_processed}")
    logger.info(f"Tokens Used: {output.total_tokens_used}")
    
    # Access metrics for cost tracking
    if result.metrics:
        logger.info(f"Total Cost: ${result.metrics.total_cost:.4f}")
else:
    logger.error(f"Agent failed: {result.error}")
```

### Global Configuration

Global configuration management with environment auto-detection:

```python
from dopeagents import get_config, set_config, DopeAgentsConfig

# Get the global config
config = get_config()
model = config.default_model
cost_tracking_enabled = config.enable_cost_tracking

# Or create and set custom config
custom_config = DopeAgentsConfig(
    default_model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
    enable_cost_tracking=True,
    max_cost_per_call=10.00,
    max_retries=3,
    log_level="DEBUG"
)
set_config(custom_config)
```

Configuration reading order:
1. Explicit environment variables (e.g., `GROQ_API_KEY`, `NVIDIA_NIM_API_KEY`)
2. `DOPEAGENTS_*` prefixed environment variables
3. Defaults from `DopeAgentsConfig`

The framework auto-detects the active provider based on available API keys.

### Logging

Professional, colorful logging:

```python
from dopeagents import get_logger

logger = get_logger(__name__)

# Output is automatically colored in dev, plain in production
logger.info("Processing request", extra={"request_id": "abc123"})
```

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/yourusername/DopeAgents.git
cd DopeAgents

# Install with uv (recommended - includes all optional dependencies)
uv sync --all-extras

# Or with pip + venv (minimal dev setup)
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -e ".[dev]"
```

### Setting Up Pre-commit Hooks

Pre-commit hooks automatically run type checking, linting, and formatting before every commit:

```bash
# Install Git hooks (one-time setup)
pre-commit install

# (Optional) Run all checks on all files to verify setup
pre-commit run --all-files
```

Now mypy, black, isort, and ruff will run automatically before each commit. To skip (not recommended): `git commit --no-verify`

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=dopeagents

# Specific test file
pytest tests/test_config.py

# Watch mode (if pytest-watch installed)
ptw
```

### Linting and Type Checking

```bash
# Type checking
mypy dopeagents

# Linting
ruff check dopeagents

# Format
ruff format dopeagents

# Import sorting
isort dopeagents
```

## Optional Dependencies

DopeAgents provides optional integrations for extended functionality:

```bash
# Research capabilities (content extraction, web search)
uv pip install dopeagents[research]

# Disk-based caching
uv pip install dopeagents[cache]

# OpenTelemetry observability
uv pip install dopeagents[otel]

# Model Context Protocol (MCP) server
uv pip install dopeagents[mcp]

# LangChain integration
uv pip install dopeagents[langchain]

# CrewAI integration
uv pip install dopeagents[crewai]

# AutoGen integration
uv pip install dopeagents[autogen]

# OpenAI provider
uv pip install dopeagents[providers]

# All optional dependencies
uv pip install dopeagents[all]
```

## Architecture Highlights

### Error Handling

DopeAgents provides a comprehensive error hierarchy:

```python
from dopeagents import (
    DopeAgentsError,           # Base exception
    ExtractionError,
    ExtractionValidationError,
    CostError,
    BudgetExceededError,
    ContractError,
)
```

### Cost Management

Track and limit API spending at multiple levels:

- **Per-call**: `DOPEAGENTS_MAX_COST_PER_CALL`
- **Per-step**: `DOPEAGENTS_MAX_COST_PER_STEP`
- **Global**: `DOPEAGENTS_MAX_COST_GLOBAL`

### Resilience

Built-in retry logic with exponential backoff:

```bash
DOPEAGENTS_ENABLE_RETRY=true
DOPEAGENTS_MAX_RETRIES=3
DOPEAGENTS_RETRY_BASE_DELAY_SECONDS=1.0
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure tests pass: `pytest`
5. Check type safety: `mypy dopeagents`
6. Submit a pull request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: See [docs/](docs/) for detailed guides
- **Issues**: [GitHub Issues](https://github.com/yourusername/DopeAgents/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/DopeAgents/discussions)

## Acknowledgments

Built with:
- [Pydantic](https://docs.pydantic.dev/) for data validation
- [LangGraph](https://python.langchain.com/langgraph/) for workflow orchestration
- [LiteLLM](https://github.com/BerriAI/litellm) for LLM integration
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output
- [Instructor](https://github.com/jxnl/instructor) for structured LLM outputs

---

**Built with 🔥 by the DopeAgents team**
