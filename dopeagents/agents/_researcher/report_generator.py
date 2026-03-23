"""Generate structured research reports in multiple formats."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ReportFormat(StrEnum):
    MARKDOWN = "markdown"
    HTML = "html"
    JSON_STRUCTURED = "json"
    EXECUTIVE_SUMMARY = "executive_summary"
    ACADEMIC = "academic"


class Citation(BaseModel):
    """A properly formatted citation."""

    index: int
    title: str
    authors: list[str] = Field(default_factory=list)
    url: str
    domain: str = ""
    published_date: str | None = None
    doi: str | None = None
    citation_count: int | None = None
    credibility_score: float = 0.0
    access_date: str = Field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d"))

    def format_apa(self) -> str:
        """Format as APA-style citation."""
        author_str = ", ".join(self.authors[:3]) if self.authors else "Unknown Author"
        if len(self.authors) > 3:
            author_str += " et al."
        year = self.published_date[:4] if self.published_date else "n.d."
        base = f"{author_str} ({year}). {self.title}."
        if self.doi:
            return f"{base} https://doi.org/{self.doi}"
        return f"{base} Retrieved from {self.url}"

    def format_inline(self) -> str:
        """Format as inline markdown citation."""
        return f"[{self.index}]"

    def format_footnote(self) -> str:
        """Format as footnote entry."""
        return f"[{self.index}]: {self.format_apa()}"


class ReportSection(BaseModel):
    """A section within the research report."""

    heading: str
    content: str
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.5, description="How well-supported this section is"
    )
    citation_indices: list[int] = Field(default_factory=list)
    is_controversial: bool = False


class StructuredReport(BaseModel):
    """A fully structured research report."""

    title: str
    query: str
    executive_summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    controversies: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    information_gaps: list[str] = Field(default_factory=list)
    methodology_note: str = ""
    citations: list[Citation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class _LLMReportStructure(BaseModel):
    """What the LLM returns for report structuring."""

    title: str = Field(description="Concise research report title")
    executive_summary: str = Field(description="2-3 paragraph executive summary for non-experts")
    sections: list[dict[str, Any]] = Field(
        description="Report sections with heading, content, citation_indices, confidence, is_controversial"
    )
    key_findings: list[str] = Field(description="Top 5 key findings, ranked by importance")
    controversies: list[str] = Field(
        default_factory=list, description="Areas where sources disagree"
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations of this research (source gaps, bias, etc.)",
    )


class ReportGenerator:
    """Generates structured, multi-format research reports from raw synthesis + metadata."""

    def __init__(self, extract_fn: Any = None):
        self._extract = extract_fn

    def build_citations(
        self,
        search_results: list[dict[str, Any]],
        credibility_scores: list[dict[str, Any]],
    ) -> list[Citation]:
        """Build properly formatted citations from search results."""
        cred_map = {s["url"]: s for s in credibility_scores}
        citations = []

        for i, result in enumerate(search_results):
            cred = cred_map.get(result["url"], {})
            citations.append(
                Citation(
                    index=i + 1,
                    title=result.get("title", "Untitled"),
                    authors=result.get("authors", []),
                    url=result["url"],
                    domain=result.get("domain", ""),
                    published_date=result.get("published_date"),
                    doi=result.get("doi"),
                    citation_count=result.get("citation_count"),
                    credibility_score=cred.get("overall", 0.0),
                )
            )

        return citations

    def structure_report(
        self,
        synthesis: str,
        key_findings: list[str],
        query: str,
        citations: list[Citation],
        claim_clusters: list[dict[str, Any]],
        information_gaps: list[str],
        model: Any = None,
    ) -> StructuredReport:
        """Use LLM to structure the synthesis into a proper report."""
        if self._extract is None:
            return StructuredReport(
                title=f"Research Report: {query}",
                query=query,
                executive_summary=synthesis[:500],
                sections=[ReportSection(heading="Findings", content=synthesis)],
                key_findings=key_findings,
                information_gaps=information_gaps,
                citations=citations,
            )

        citation_list = "\n".join(
            f"[{c.index}] {c.title} ({c.domain}) - credibility: {c.credibility_score:.2f}"
            for c in citations
        )

        clusters_text = "\n".join(
            f"- {c['representative_claim']} (agreement: {c['agreement_score']})"
            for c in claim_clusters
        )

        result = self._extract(
            response_model=_LLMReportStructure,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Structure the research synthesis into a well-organized report. "
                        "Create logical sections, write an executive summary, and identify "
                        "limitations. Reference citations by their [index] numbers. "
                        "Be honest about what the research does and does not establish."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Synthesis:\n{synthesis}\n\n"
                        f"Available Citations:\n{citation_list}\n\n"
                        f"Claim Analysis:\n{clusters_text}\n\n"
                        f"Known Gaps:\n" + "\n".join(f"- {g}" for g in information_gaps)
                    ),
                },
            ],
            model=model,
            allow_fallback=True,
        )

        sections = [
            ReportSection(
                heading=s.get("heading", "Section"),
                content=s.get("content", ""),
                confidence=s.get("confidence", 0.5),
                citation_indices=s.get("citation_indices", []),
                is_controversial=s.get("is_controversial", False),
            )
            for s in result.sections
        ]

        return StructuredReport(
            title=result.title,
            query=query,
            executive_summary=result.executive_summary,
            sections=sections,
            key_findings=result.key_findings,
            controversies=result.controversies,
            limitations=result.limitations,
            information_gaps=information_gaps,
            methodology_note=(
                "This report was generated using automated search across Wikipedia, "
                "DuckDuckGo, Semantic Scholar, arXiv, and CrossRef. Sources were scored "
                "for credibility using domain authority, citation count, and recency signals. "
                "Claims were extracted and cross-referenced across sources."
            ),
            citations=citations,
            metadata={
                "generated_at": datetime.now(UTC).isoformat(),
                "total_sources": len(citations),
                "avg_credibility": (
                    sum(c.credibility_score for c in citations) / len(citations)
                    if citations
                    else 0.0
                ),
            },
        )

    def render(self, report: StructuredReport, fmt: ReportFormat) -> str:
        """Render a structured report to the requested format."""
        renderers = {
            ReportFormat.MARKDOWN: self._render_markdown,
            ReportFormat.HTML: self._render_html,
            ReportFormat.JSON_STRUCTURED: self._render_json,
            ReportFormat.EXECUTIVE_SUMMARY: self._render_executive,
            ReportFormat.ACADEMIC: self._render_academic,
        }
        return renderers[fmt](report)

    def _render_markdown(self, report: StructuredReport) -> str:
        lines = [
            f"# {report.title}",
            "",
            f"*Query: {report.query}*",
            "",
            "## Executive Summary",
            "",
            report.executive_summary,
            "",
        ]

        self._md_key_findings(lines, report.key_findings)
        self._md_sections(lines, report.sections)
        self._md_bullet_section(lines, "Areas of Disagreement", report.controversies, "⚠️ ")
        self._md_bullet_section(lines, "Limitations", report.limitations)
        self._md_bullet_section(lines, "Information Gaps", report.information_gaps)

        if report.methodology_note:
            lines.extend(["## Methodology", "", report.methodology_note, ""])

        lines.extend(["## References", ""])
        for citation in sorted(report.citations, key=lambda c: c.index):
            lines.append(citation.format_footnote())
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _md_key_findings(lines: list[str], findings: list[str]) -> None:
        if findings:
            lines.extend(["## Key Findings", ""])
            for i, finding in enumerate(findings, 1):
                lines.append(f"{i}. {finding}")
            lines.append("")

    @staticmethod
    def _md_sections(lines: list[str], sections: list[ReportSection]) -> None:
        for section in sections:
            marker = " ⚠️" if section.is_controversial else ""
            confidence_bar = "●" * int(section.confidence * 5) + "○" * (
                5 - int(section.confidence * 5)
            )
            lines.extend(
                [
                    f"## {section.heading}{marker}",
                    f"*Confidence: {confidence_bar} ({section.confidence:.0%})*",
                    "",
                    section.content,
                    "",
                ]
            )

    @staticmethod
    def _md_bullet_section(
        lines: list[str], heading: str, items: list[str], prefix: str = ""
    ) -> None:
        if items:
            lines.extend([f"## {heading}", ""])
            for item in items:
                lines.append(f"- {prefix}{item}")
            lines.append("")

    def _render_html(self, report: StructuredReport) -> str:
        md = self._render_markdown(report)
        try:
            import markdown  # type: ignore[import-untyped]

            body = markdown.markdown(md, extensions=["tables", "fenced_code"])
        except ImportError:
            body = f"<pre>{md}</pre>"

        title = report.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        generated_at = report.metadata.get("generated_at", "N/A")
        total_sources = report.metadata.get("total_sources", 0)
        avg_cred = report.metadata.get("avg_credibility", 0)

        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '    <meta charset="UTF-8">\n'
            '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"    <title>{title}</title>\n"
            "    <style>\n"
            "        body { max-width: 800px; margin: 2rem auto; font-family: Georgia, serif; "
            "line-height: 1.6; padding: 0 1rem; }\n"
            "        h1 { border-bottom: 2px solid #333; padding-bottom: 0.5rem; }\n"
            "        h2 { color: #2c3e50; margin-top: 2rem; }\n"
            "        blockquote { border-left: 3px solid #3498db; padding-left: 1rem; color: #555; }\n"
            "        .metadata { color: #888; font-size: 0.9em; }\n"
            "    </style>\n"
            "</head>\n<body>\n"
            f"{body}\n"
            "<hr>\n"
            f'<p class="metadata">Generated: {generated_at} | '
            f"Sources: {total_sources} | "
            f"Avg Credibility: {avg_cred:.0%}</p>\n"
            "</body>\n</html>"
        )

    def _render_json(self, report: StructuredReport) -> str:
        return report.model_dump_json(indent=2)

    def _render_executive(self, report: StructuredReport) -> str:
        lines = [
            f"# {report.title}",
            "",
            report.executive_summary,
            "",
            "**Key Findings:**",
        ]
        for i, f in enumerate(report.key_findings, 1):
            lines.append(f"  {i}. {f}")

        if report.controversies:
            lines.extend(["", "**Note:** " + "; ".join(report.controversies[:2])])

        return "\n".join(lines)

    def _render_academic(self, report: StructuredReport) -> str:
        lines = [
            f"# {report.title}",
            "",
            "## Abstract",
            "",
            report.executive_summary,
            "",
        ]

        for section in report.sections:
            lines.extend([f"## {section.heading}", "", section.content, ""])

        lines.extend(["## Discussion", ""])
        if report.limitations:
            lines.append("### Limitations")
            lines.append("")
            for lim in report.limitations:
                lines.append(f"- {lim}")
            lines.append("")

        lines.extend(["## References", ""])
        for citation in sorted(report.citations, key=lambda c: c.index):
            lines.append(f"{citation.index}. {citation.format_apa()}")

        return "\n".join(lines)
