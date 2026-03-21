"""
DeepSummarizer capability test suite.

Covers:
  1. Bullets style  -- multi-paragraph technical article
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

# Silence LiteLLM's "Give Feedback / Provider List" banner noise
os.environ.setdefault("LITELLM_LOG", "ERROR")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)

from dopeagents.agents import DeepSummarizer, DeepSummarizerInput, DeepSummarizerOutput
from dopeagents.core.types import AgentResult

# Must be set AFTER litellm is imported (transitively via agents)
import litellm  # noqa: E402
litellm.suppress_debug_info = True
litellm.set_verbose = False

# ---------------------------------------------------------------------------
# Sample texts
# ---------------------------------------------------------------------------

PAGEINDEX_ARTICLE = """\
Long documents (10-Ks, contracts, regulatory filings, manuals) are where retrieval
systems usually fall short. PageIndex is a fresh approach: instead of chopping text
into fixed chunks and searching an embedding space, it builds a hierarchical tree
(intelligent table of contents) and uses LLM reasoning to search that tree in a
human-like, traceable way to retrieve information.

What PageIndex actually does

Tree generation: ingest a document (PDF/HTML) and convert it into a hierarchical tree
of nodes (sections/subsections/pages).

LLM-driven tree search: instead of nearest-neighbor vectors, an LLM reasons about
which nodes to visit, returns node IDs, and the content from those nodes is used to
answer queries.

No chunking, no vector DB: this reduces chunking artifacts and the vibe-retrieval
failure modes of vector similarity.  For long, structured documents context spans
sections and cross-references; a tree + reasoning approach preserves and exploits
that structure so answers are more explainable and often more accurate.

High-level architecture

  1. Ingest document -> parse headings, tables, lists -> produce a tree index.
  2. Query arrives -> PageIndex asks an LLM to search the tree and return node IDs.
  3. Extract text from picked nodes -> feed to LLM to generate the final answer
     along with a trace (node references). Optionally follow references and iterate.

When to use PageIndex:
  - Very long, structured documents (annual reports, contracts, clinical protocols).
  - Queries that require following in-document cross-references (appendices, tables).
  - Use-cases needing auditability -- PageIndex returns node references and a
    reasoning trace.
"""

TRANSFORMER_OVERVIEW = """\
The Transformer architecture, introduced in Attention Is All You Need (Vaswani et al. 2017),
replaced recurrent networks with a fully attention-based model. Its two core components are
multi-head self-attention, which lets each token attend to every other token simultaneously,
and position-wise feed-forward layers that apply non-linear transformations independently at
each position.

Scaled dot-product attention computes compatibility scores between queries and keys, normalises
them with a softmax, then uses those weights to aggregate values.  Running several attention
heads in parallel allows the model to jointly attend to information from different representation
sub-spaces.

The encoder maps an input sequence to continuous representations.  The decoder generates the
output sequence one token at a time using masked self-attention and cross-attention to the
encoder.  Residual connections and layer normalisation surround each sub-layer.

Positional encodings inject sequence order information because self-attention is
permutation-invariant.

Pre-trained Transformer variants (BERT, GPT, T5, LLaMA) have set state-of-the-art results
across NLP, vision, audio, and multimodal tasks, making the architecture the dominant paradigm
in modern deep learning.
"""

SHORT_CLIMATE_BLURB = """\
Global average surface temperature has risen approximately 1.2 degrees Celsius above
pre-industrial levels as of 2023, driven primarily by greenhouse-gas emissions from
fossil-fuel combustion and deforestation. The Intergovernmental Panel on Climate Change
projects further warming of 1.5-4.5 degrees Celsius by 2100 depending on emission
trajectories. Consequences include more frequent extreme weather events, sea-level rise
threatening coastal communities, and disruption of agricultural systems worldwide.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SEPARATOR = "-" * 60


@dataclass
class TestCase:
    name: str
    input: DeepSummarizerInput
    assertions: list[tuple[str, Any]]


def run_case(agent: DeepSummarizer, case: TestCase) -> bool:
    print(f"\n{'=' * 60}")
    print(f"TEST: {case.name}")
    print(SEPARATOR)

    result: AgentResult[DeepSummarizerOutput] = agent.run(case.input)

    if not result.success or result.output is None:
        print(f"  [ERROR] Agent run failed: {result.error}")
        # Mark all assertions as failed
        print("  Assertions:")
        for label, _ in case.assertions:
            print(f"    [FAIL] {label}")
        return False

    out = result.output

    print(f"  style             : {case.input.style}")
    print(f"  focus             : {case.input.focus!r}")
    print(f"  chunks_processed  : {out.chunks_processed}")
    print(f"  refinement_rounds : {out.refinement_rounds}")
    print(f"  quality_score     : {out.quality_score:.2f}")
    print(f"  word_count        : {out.word_count}")
    print(f"  truncated         : {out.truncated}")
    print(f"\n  key_points ({len(out.key_points)}):")
    for i, kp in enumerate(out.key_points, 1):
        print(f"    {i}. {kp}")
    print(f"\n  summary:\n{out.summary}\n")

    all_passed = True
    print("  Assertions:")
    for label, check in case.assertions:
        try:
            passed = check(out)
        except Exception as exc:
            passed = False
            label = f"{label} [exception: {exc}]"
        status = PASS if passed else FAIL
        print(f"    [{status}] {label}")
        if not passed:
            all_passed = False

    return all_passed


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def build_cases() -> list[TestCase]:
    return [
        TestCase(
            name="1 . Bullets style -- technical article",
            input=DeepSummarizerInput(
                text=PAGEINDEX_ARTICLE,
                max_length=800,
                style="bullets",
            ),
            assertions=[
                ("quality_score >= 0.4", lambda o: o.quality_score >= 0.4),
                ("chunks_processed >= 1", lambda o: o.chunks_processed >= 1),
                ("refinement_rounds >= 0", lambda o: o.refinement_rounds >= 0),
                ("summary is non-empty", lambda o: len(o.summary.strip()) > 0),
                ("key_points is non-empty", lambda o: len(o.key_points) > 0),
                ("word_count > 0", lambda o: o.word_count > 0),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    agent = DeepSummarizer()
    cases = build_cases()

    results: list[tuple[str, bool]] = []
    for case in cases:
        passed = run_case(agent, case)
        results.append((case.name, passed))

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(SEPARATOR)
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    for name, passed in results:
        status = PASS if passed else FAIL
        print(f"  [{status}] {name}")
    print(SEPARATOR)
    print(f"  {passed_count}/{total} tests passed")
    print("=" * 60)

    if passed_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
