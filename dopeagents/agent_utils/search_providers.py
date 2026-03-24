"""Real search providers using free public APIs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result from any provider."""

    title: str
    url: str
    snippet: str
    source_provider: str
    domain: str = ""
    published_date: str | None = None
    authors: list[str] = field(default_factory=list)
    doi: str | None = None
    citation_count: int | None = None
    content_type: str = "web"  # web, academic, news, encyclopedia
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, calls_per_second: float = 1.0) -> None:
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


class DiskCache:
    """Simple disk-based cache for search results (avoids re-hitting APIs)."""

    def __init__(self, cache_dir: str = ".research_cache", ttl_hours: int = 24) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_hours * 3600

    def _key(self, provider: str, query: str) -> str:
        h = hashlib.sha256(f"{provider}:{query}".encode()).hexdigest()[:16]
        return h

    def get(self, provider: str, query: str) -> list[dict[str, Any]] | None:
        path = self._cache_dir / f"{self._key(provider, query)}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if time.time() - data["timestamp"] > self._ttl_seconds:
            path.unlink()
            return None
        return data["results"]  # type: ignore[no-any-return]

    def set(self, provider: str, query: str, results: list[dict[str, Any]]) -> None:
        path = self._cache_dir / f"{self._key(provider, query)}.json"
        path.write_text(
            json.dumps(
                {
                    "timestamp": time.time(),
                    "provider": provider,
                    "query": query,
                    "results": results,
                }
            )
        )


# ── Abstract Base ──────────────────────────────────────────────────────


class SearchProvider(ABC):
    """Base class for all search providers."""

    name: str
    content_type: str = "web"
    _default_user_agent = (
        "Mozilla/5.0 (compatible; DopeAgents/3.1; +https://github.com/cosmos-127/dopeagents)"
    )

    def __init__(self, rate_limiter: RateLimiter | None = None) -> None:
        self._rate_limiter = rate_limiter or RateLimiter(1.0)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": self._default_user_agent,
                    "Accept": "application/json, text/plain, */*",
                },
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]: ...


# ── Wikipedia ──────────────────────────────────────────────────────────


class WikipediaProvider(SearchProvider):
    """Wikipedia search API — completely free, no key required."""

    name = "wikipedia"
    content_type = "encyclopedia"

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        search_resp = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
                "srprop": "snippet|timestamp",
            },
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()

        results: list[SearchResult] = []
        for item in search_data.get("query", {}).get("search", []):
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            title = item["title"]
            results.append(
                SearchResult(
                    title=title,
                    url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    snippet=snippet,
                    source_provider=self.name,
                    domain="en.wikipedia.org",
                    content_type=self.content_type,
                    published_date=item.get("timestamp"),
                    raw_metadata=item,
                )
            )
        return results


# ── DuckDuckGo ─────────────────────────────────────────────────────────


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo Instant Answer API — free, no key required."""

    name = "duckduckgo"
    content_type = "web"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        results: list[SearchResult] = []

        client = await self._get_client()
        try:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("AbstractText"):
                results.append(
                    SearchResult(
                        title=data.get("Heading", query),
                        url=data.get("AbstractURL", ""),
                        snippet=data["AbstractText"][:500],
                        source_provider=self.name,
                        domain=data.get("AbstractSource", ""),
                        content_type="web",
                    )
                )

            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and "FirstURL" in topic:
                    results.append(
                        SearchResult(
                            title=topic.get("Text", "")[:100],
                            url=topic["FirstURL"],
                            snippet=topic.get("Text", "")[:500],
                            source_provider=self.name,
                            content_type="web",
                        )
                    )
        except Exception as e:
            logger.warning("DuckDuckGo instant answer failed: %s", e)

        # Fall back to duckduckgo-search library if available and we need more results
        if len(results) < max_results:
            try:
                from duckduckgo_search import DDGS

                loop = asyncio.get_event_loop()
                remaining = max_results - len(results)
                ddg_results = await loop.run_in_executor(
                    None, lambda: list(DDGS().text(query, max_results=remaining))
                )
                for r in ddg_results:
                    results.append(
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", r.get("link", "")),
                            snippet=r.get("body", r.get("snippet", "")),
                            source_provider=self.name,
                            content_type="web",
                        )
                    )
            except ImportError:
                logger.info("duckduckgo-search not installed; using instant answer API only")
            except Exception as e:
                logger.warning("duckduckgo-search fallback failed: %s", e)

        return results[:max_results]


# ── Semantic Scholar ───────────────────────────────────────────────────


class SemanticScholarProvider(SearchProvider):
    """Semantic Scholar API — free, 214M+ papers, AI TLDRs, citation graphs.

    See: https://api.semanticscholar.org/api-docs/
    Rate limit: 1000 req/sec shared (unauthenticated), 1 RPS with free API key
    Best for: CS, AI, biomedical, all disciplines
    """

    name = "semantic_scholar"
    content_type = "academic"

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(rate_limiter=RateLimiter(0.5), **kwargs)
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        headers: dict[str, str] = {}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        resp = await client.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": min(max_results, 100),
                "fields": "title,abstract,url,year,authors,citationCount,externalIds,tldr",
            },
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for paper in data.get("data", []):
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            ext_ids = paper.get("externalIds", {}) or {}
            doi = ext_ids.get("DOI")
            url = paper.get("url", "")
            if not url and doi:
                url = f"https://doi.org/{doi}"

            tldr = paper.get("tldr")
            snippet = ""
            if tldr and isinstance(tldr, dict):
                snippet = tldr.get("text", "")
            if not snippet:
                snippet = (paper.get("abstract") or "")[:500]

            results.append(
                SearchResult(
                    title=paper.get("title", ""),
                    url=url,
                    snippet=snippet,
                    source_provider=self.name,
                    domain="semanticscholar.org",
                    authors=authors,
                    doi=doi,
                    citation_count=paper.get("citationCount"),
                    published_date=str(paper.get("year", "")) or None,
                    content_type="academic",
                    raw_metadata=paper,
                )
            )
        return results


# ── OpenAlex ──────────────────────────────────────────────────────────


class OpenAlexProvider(SearchProvider):
    """OpenAlex API — 250M+ scholarly works, free API key required (free).

    See: https://docs.openalex.org
    Rate limit: 100k credits/day with free key, 100/day without
    Best for: Broad academic search, citation analysis, institutional data
    """

    name = "openalex"
    content_type = "academic"

    def __init__(self, email: str | None = None, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(rate_limiter=RateLimiter(1.0), **kwargs)
        self._email = email
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        params: dict[str, Any] = {
            "search": query,
            "per_page": min(max_results, 50),
        }
        if self._email:
            params["mailto"] = self._email
        if self._api_key:
            params["api_key"] = self._api_key

        resp = await client.get("https://api.openalex.org/works", params=params)
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for work in data.get("results", []):
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                name = author.get("display_name", "")
                if name:
                    authors.append(name)

            doi = work.get("doi", "")
            url = doi if doi else work.get("id", "")

            snippet = ""
            abstract_inv = work.get("abstract_inverted_index")
            if abstract_inv and isinstance(abstract_inv, dict):
                word_positions: list[tuple[int, str]] = []
                for word, positions in abstract_inv.items():
                    for pos in positions:
                        word_positions.append((pos, word))
                word_positions.sort()
                snippet = " ".join(w for _, w in word_positions)[:500]

            results.append(
                SearchResult(
                    title=work.get("title", "") or "",
                    url=url,
                    snippet=snippet,
                    source_provider=self.name,
                    domain="openalex.org",
                    authors=authors,
                    doi=doi.replace("https://doi.org/", "") if doi else None,
                    citation_count=work.get("cited_by_count"),
                    published_date=work.get("publication_date"),
                    content_type="academic",
                    raw_metadata={
                        "type": work.get("type"),
                        "open_access": work.get("open_access", {}),
                        "source": work.get("primary_location", {}).get("source", {}),
                    },
                )
            )
        return results


# ── PubMed ────────────────────────────────────────────────────────────


class PubMedProvider(SearchProvider):
    """PubMed Central API — free, open, 9M+ biomedical articles.

    See: https://www.ncbi.nlm.nih.gov/pmc/
    Rate limit: Reasonable (~3 req/sec)
    Best for: Medical, health, biomedical research
    """

    name = "pubmed"
    content_type = "academic"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(rate_limiter=RateLimiter(0.3), **kwargs)

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        # Step 1: Search for PMCIDs
        try:
            search_resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pmc",
                    "term": query,
                    "retmax": min(max_results, 50),
                    "rettype": "json",
                },
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()

            pmcids = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmcids:
                return []

            # Step 2: Fetch summaries for each article
            summary_resp = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={
                    "db": "pmc",
                    "id": ",".join(pmcids[: min(max_results, 20)]),
                    "rettype": "json",
                },
            )
            summary_resp.raise_for_status()
            summary_data = summary_resp.json()
        except Exception as e:
            logger.warning("PubMed search failed: %s", e)
            return []

        results: list[SearchResult] = []
        for item in summary_data.get("result", {}).values():
            if isinstance(item, dict) and "title" in item:
                # Build author list
                authors = []
                for author in item.get("authors", []):
                    if isinstance(author, dict):
                        name = author.get("name", "")
                        if name:
                            authors.append(name)

                pmcid = item.get("uid", "")
                pub_date = item.get("pubdate", "")

                snippet = item.get("abstract", "") or item.get("title", "")
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
                        snippet=snippet[:500],
                        source_provider=self.name,
                        domain="ncbi.nlm.nih.gov",
                        authors=authors,
                        published_date=pub_date,
                        content_type="academic",
                        raw_metadata=item,
                    )
                )
        return results


# ── CORE ──────────────────────────────────────────────────────────────


class COREProvider(SearchProvider):
    """CORE API — aggregates 200M+ research outputs from 10k+ sources.

    See: https://api.core.ac.uk/v3
    Free tier: 100 req/month, or use API key for more.
    Rate limit: Reasonable
    """

    name = "core"
    content_type = "academic"

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(rate_limiter=RateLimiter(0.3), **kwargs)
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # CORE API v3 search endpoint
        # See: https://api.core.ac.uk/v3/docs
        resp = await client.get(
            "https://api.core.ac.uk/v3/search/works",
            params={
                "q": query,
                "limit": min(max_results, 50),
            },
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for work in data.get("results", []):
            authors = [author.get("name", "") for author in work.get("authors", [])]

            # Get the best available URL
            url = ""
            if work.get("links"):
                url = work["links"][0].get("url", "")
            if not url and work.get("id"):
                url = f"https://core.ac.uk/display/{work['id']}"

            results.append(
                SearchResult(
                    title=work.get("title", ""),
                    url=url,
                    snippet=(work.get("abstract", "") or "")[:500],
                    source_provider=self.name,
                    domain="core.ac.uk",
                    authors=authors,
                    citation_count=work.get("citationCount"),
                    content_type="academic",
                    published_date=work.get("publishedDate"),
                    raw_metadata=work,
                )
            )
        return results


# ── arXiv ──────────────────────────────────────────────────────────────


class ArxivProvider(SearchProvider):
    """arXiv API — completely free, no key required."""

    name = "arxiv"
    content_type = "academic"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        resp = await client.get(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        resp.raise_for_status()

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)

        results: list[SearchResult] = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
            published = entry.findtext("atom:published", "", ns) or ""

            url = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    url = link.get("href", "")
                    break
            if not url:
                url = entry.findtext("atom:id", "", ns) or ""

            authors = [
                author.findtext("atom:name", "", ns) or ""
                for author in entry.findall("atom:author", ns)
            ]

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=summary[:500],
                    source_provider=self.name,
                    domain="arxiv.org",
                    authors=authors,
                    published_date=published[:10] if published else None,
                    content_type="academic",
                )
            )
        return results


# ── CrossRef ───────────────────────────────────────────────────────────


class CrossRefProvider(SearchProvider):
    """CrossRef API — free, no key required (polite pool with email)."""

    name = "crossref"
    content_type = "academic"

    def __init__(self, email: str | None = None, **kwargs: Any) -> None:
        super().__init__(rate_limiter=RateLimiter(0.5), **kwargs)
        self._email = email

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        params: dict[str, Any] = {
            "query": query,
            "rows": max_results,
            "sort": "relevance",
            "order": "desc",
        }
        if self._email:
            params["mailto"] = self._email

        resp = await client.get("https://api.crossref.org/works", params=params)
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for item in data.get("message", {}).get("items", []):
            title_parts = item.get("title", [])
            title = title_parts[0] if title_parts else ""
            abstract = re.sub(r"<[^>]+>", "", item.get("abstract", ""))

            authors: list[str] = []
            for author in item.get("author", []):
                name_parts = [author.get("given", ""), author.get("family", "")]
                authors.append(" ".join(p for p in name_parts if p))

            doi = item.get("DOI", "")
            url = f"https://doi.org/{doi}" if doi else item.get("URL", "")

            date_parts = item.get("published-print", item.get("published-online", {}))
            date_arr = date_parts.get("date-parts", [[]])[0] if date_parts else []
            pub_date = "-".join(str(d) for d in date_arr) if date_arr else None

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=abstract[:500],
                    source_provider=self.name,
                    domain="crossref.org",
                    authors=authors,
                    doi=doi or None,
                    citation_count=item.get("is-referenced-by-count"),
                    published_date=pub_date,
                    content_type="academic",
                    raw_metadata={"container_title": item.get("container-title", [])},
                )
            )
        return results


# ── Unpaywall ──────────────────────────────────────────────────────────


class UnpaywallProvider(SearchProvider):
    """Unpaywall API — finds open access full-text for DOIs. Free, no key.

    See: https://unpaywall.org/products/api
    Rate limit: 100,000 calls/day
    Note: DOI-lookup provider, not a search provider. Use ``find_open_access``
    to enrich results from CrossRef/OpenAlex with free full-text PDF links.
    """

    name = "unpaywall"
    content_type = "academic"

    def __init__(self, email: str = "user@example.com", **kwargs: Any) -> None:
        super().__init__(rate_limiter=RateLimiter(10.0), **kwargs)
        self._email = email

    async def search(self, _query: str, _max_results: int = 10) -> list[SearchResult]:
        return []

    async def find_open_access(self, doi: str) -> SearchResult | None:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        resp = await client.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": self._email},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()

        best_oa = data.get("best_oa_location") or {}
        pdf_url = best_oa.get("url_for_pdf") or best_oa.get("url", "")
        if not pdf_url:
            return None

        return SearchResult(
            title=data.get("title", ""),
            url=pdf_url,
            snippet=f"Open access ({data.get('oa_status', 'unknown')}) via {best_oa.get('host_type', 'unknown')}",
            source_provider=self.name,
            domain=urlparse(pdf_url).netloc,
            doi=doi,
            published_date=data.get("published_date"),
            content_type="academic",
            raw_metadata=data,
        )


# ── GNews ──────────────────────────────────────────────────────────────


class GNewsProvider(SearchProvider):
    """GNews API — free tier: 100 req/day, 10 articles per search.

    See: https://gnews.io/docs/v4
    Free API key required.
    Best for: Current news, events, trending topics
    """

    name = "gnews"
    content_type = "news"

    def __init__(self, api_key: str = "", **kwargs: Any) -> None:
        super().__init__(rate_limiter=RateLimiter(0.2), **kwargs)
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        if not self._api_key:
            return []

        await self._rate_limiter.acquire()
        client = await self._get_client()

        resp = await client.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": query,
                "lang": "en",
                "max": min(max_results, 10),
                "apikey": self._api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for article in data.get("articles", []):
            results.append(
                SearchResult(
                    title=article.get("title", ""),
                    url=article.get("url", ""),
                    snippet=article.get("description", "")[:500],
                    source_provider=self.name,
                    domain=article.get("source", {}).get("url", ""),
                    published_date=article.get("publishedAt"),
                    content_type="news",
                    raw_metadata=article,
                )
            )
        return results


# ── NYT ────────────────────────────────────────────────────────────────


class NYTimesProvider(SearchProvider):
    """NYT Article Search API — free key, 500 req/day.

    See: https://developer.nytimes.com/
    Best for: Quality journalism, historical news, current events
    """

    name = "nytimes"
    content_type = "news"

    def __init__(self, api_key: str = "", **kwargs: Any) -> None:
        super().__init__(rate_limiter=RateLimiter(0.2), **kwargs)
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        if not self._api_key:
            return []

        await self._rate_limiter.acquire()
        client = await self._get_client()

        resp = await client.get(
            "https://api.nytimes.com/svc/search/v2/articlesearch.json",
            params={"q": query, "api-key": self._api_key},
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for doc in data.get("response", {}).get("docs", [])[:max_results]:
            snippet = doc.get("abstract", "") or doc.get("lead_paragraph", "")
            results.append(
                SearchResult(
                    title=doc.get("headline", {}).get("main", ""),
                    url=doc.get("web_url", ""),
                    snippet=snippet[:500],
                    source_provider=self.name,
                    domain="nytimes.com",
                    published_date=doc.get("pub_date"),
                    content_type="news",
                    raw_metadata=doc,
                )
            )
        return results


# ── Open Library ───────────────────────────────────────────────────────


class OpenLibraryProvider(SearchProvider):
    """Open Library API — free, no key, 20M+ books.

    See: https://openlibrary.org/developers/api
    Best for: Book references, author info, historical texts
    """

    name = "openlibrary"
    content_type = "books"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        resp = await client.get(
            "https://openlibrary.org/search.json",
            params={
                "q": query,
                "limit": max_results,
                "fields": "key,title,author_name,first_publish_year,subject,isbn,edition_count",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for doc in data.get("docs", [])[:max_results]:
            key = doc.get("key", "")
            authors = doc.get("author_name", [])
            subjects = doc.get("subject", [])[:5]
            snippet = f"Authors: {', '.join(authors[:3])}. " if authors else ""
            if subjects:
                snippet += f"Subjects: {', '.join(subjects)}. "
            snippet += f"First published: {doc.get('first_publish_year', 'Unknown')}. "
            snippet += f"Editions: {doc.get('edition_count', 'Unknown')}."

            results.append(
                SearchResult(
                    title=doc.get("title", ""),
                    url=f"https://openlibrary.org{key}",
                    snippet=snippet[:500],
                    source_provider=self.name,
                    domain="openlibrary.org",
                    authors=authors[:5],
                    published_date=str(doc.get("first_publish_year", "")) or None,
                    content_type="books",
                    raw_metadata=doc,
                )
            )
        return results


# ── FRED ───────────────────────────────────────────────────────────────


class FREDProvider(SearchProvider):
    """FRED API — 765,000+ economic time series. Free API key required.

    See: https://fred.stlouisfed.org/docs/api/fred/
    Best for: Economic data, GDP, inflation, employment, interest rates
    """

    name = "fred"
    content_type = "data"

    def __init__(self, api_key: str = "", **kwargs: Any) -> None:
        super().__init__(rate_limiter=RateLimiter(1.0), **kwargs)
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        if not self._api_key:
            return []

        await self._rate_limiter.acquire()
        client = await self._get_client()

        resp = await client.get(
            "https://api.stlouisfed.org/fred/series/search",
            params={
                "search_text": query,
                "api_key": self._api_key,
                "file_type": "json",
                "limit": max_results,
                "order_by": "search_rank",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for series in data.get("seriess", []):
            series_id = series.get("id", "")
            notes = (series.get("notes") or "")[:400]
            freq = series.get("frequency", "N/A")
            units = series.get("units", "N/A")
            results.append(
                SearchResult(
                    title=series.get("title", ""),
                    url=f"https://fred.stlouisfed.org/series/{series_id}",
                    snippet=f"{notes}. Frequency: {freq}. Units: {units}.",
                    source_provider=self.name,
                    domain="fred.stlouisfed.org",
                    published_date=series.get("last_updated"),
                    content_type="data",
                    raw_metadata=series,
                )
            )
        return results


# ── World Bank ─────────────────────────────────────────────────────────


class WorldBankProvider(SearchProvider):
    """World Bank Documents & Reports API — completely free, no key required.

    See: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
    Best for: International development, country-level economic/social indicators
    """

    name = "worldbank"
    content_type = "data"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        resp = await client.get(
            "https://search.worldbank.org/api/v2/wds",
            params={
                "format": "json",
                "qterm": query,
                "rows": max_results,
                "os": 0,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for doc in data.get("documents", {}).values():
            if not isinstance(doc, dict) or "display_title" not in doc:
                continue

            # abstracts may be a dict or list, not a string
            abstracts = doc.get("abstracts", "")
            if isinstance(abstracts, dict):
                abstracts = abstracts.get("en", "") or ""
            elif isinstance(abstracts, list | tuple):
                abstracts = " ".join(str(a) for a in abstracts)
            snippet = str(abstracts)[:500] if abstracts else ""

            results.append(
                SearchResult(
                    title=doc.get("display_title", ""),
                    url=doc.get("url", "") or doc.get("pdfurl", ""),
                    snippet=snippet,
                    source_provider=self.name,
                    domain="worldbank.org",
                    published_date=doc.get("docdt"),
                    content_type="data",
                    raw_metadata=doc,
                )
            )
        return results


# ── GitHub ─────────────────────────────────────────────────────────────


class GitHubSearchProvider(SearchProvider):
    """GitHub Search API — free, 10 req/min unauthenticated, 30 with token.

    See: https://docs.github.com/en/rest/search
    Best for: Code, repos, technical implementations, READMEs
    """

    name = "github"
    content_type = "code"

    def __init__(self, token: str | None = None, **kwargs: Any) -> None:
        super().__init__(rate_limiter=RateLimiter(0.15), **kwargs)
        self._token = token

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        resp = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "per_page": min(max_results, 30)},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for repo in data.get("items", []):
            results.append(
                SearchResult(
                    title=repo.get("full_name", ""),
                    url=repo.get("html_url", ""),
                    snippet=(repo.get("description", "") or "")[:500],
                    source_provider=self.name,
                    domain="github.com",
                    published_date=repo.get("updated_at"),
                    content_type="code",
                    raw_metadata={
                        "stars": repo.get("stargazers_count"),
                        "language": repo.get("language"),
                        "topics": repo.get("topics", []),
                    },
                )
            )
        return results


# ── US Government ──────────────────────────────────────────────────────


class USGovernmentProvider(SearchProvider):
    """Federal Register API — free, no key required, US regulations/rules.

    See: https://www.federalregister.gov/developers/api/v1
    Best for: Government reports, policy data, regulations
    """

    name = "us_gov"
    content_type = "government"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        resp = await client.get(
            "https://www.federalregister.gov/api/v1/documents.json",
            params={
                "conditions[term]": query,
                "per_page": min(max_results, 20),
                "order": "relevance",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for doc in data.get("results", []):
            snippet = doc.get("abstract") or doc.get("excerpt", "")
            results.append(
                SearchResult(
                    title=doc.get("title", ""),
                    url=doc.get("html_url", ""),
                    snippet=(snippet or "")[:500],
                    source_provider=self.name,
                    domain="federalregister.gov",
                    published_date=doc.get("publication_date"),
                    content_type="government",
                    raw_metadata=doc,
                )
            )
        return results


# ── DPLA ───────────────────────────────────────────────────────────────


class DPLAProvider(SearchProvider):
    """Digital Public Library of America API — free API key.

    See: https://pro.dp.la/developers/api-codex
    Best for: Historical documents, cultural heritage, primary sources
    """

    name = "dpla"
    content_type = "cultural"

    def __init__(self, api_key: str = "", **kwargs: Any) -> None:
        super().__init__(rate_limiter=RateLimiter(1.0), **kwargs)
        self._api_key = api_key

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        if not self._api_key:
            return []

        await self._rate_limiter.acquire()
        client = await self._get_client()

        resp = await client.get(
            "https://api.dp.la/v2/items",
            params={"q": query, "page_size": max_results, "api_key": self._api_key},
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for doc in data.get("docs", []):
            source_resource = doc.get("sourceResource", {})
            title_field = source_resource.get("title", [""])
            title = title_field[0] if isinstance(title_field, list) else title_field

            desc = source_resource.get("description", [""])
            snippet = desc[0] if isinstance(desc, list) else desc

            date_info = source_resource.get("date", {})
            pub_date = None
            if isinstance(date_info, dict):
                pub_date = str(date_info.get("begin", "")) or None
            elif isinstance(date_info, list) and date_info:
                pub_date = str(date_info[0].get("begin", "")) or None

            results.append(
                SearchResult(
                    title=title,
                    url=doc.get("isShownAt", ""),
                    snippet=(snippet or "")[:500],
                    source_provider=self.name,
                    domain="dp.la",
                    published_date=pub_date,
                    content_type="cultural",
                    raw_metadata=doc,
                )
            )
        return results


# ── Library of Congress ────────────────────────────────────────────────


class LibraryOfCongressProvider(SearchProvider):
    """Library of Congress API — free, no key required.

    See: https://www.loc.gov/apis/
    Best for: Historical documents, primary sources, newspapers, maps
    """

    name = "loc"
    content_type = "cultural"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        await self._rate_limiter.acquire()
        client = await self._get_client()

        resp = await client.get(
            "https://www.loc.gov/search/",
            params={"q": query, "fo": "json", "c": max_results},
        )
        resp.raise_for_status()
        data = resp.json()

        results: list[SearchResult] = []
        for item in data.get("results", [])[:max_results]:
            desc = item.get("description", [])
            snippet = desc[0] if desc else ""

            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", "") or item.get("id", ""),
                    snippet=snippet[:500],
                    source_provider=self.name,
                    domain="loc.gov",
                    published_date=item.get("date"),
                    content_type="cultural",
                    raw_metadata=item,
                )
            )
        return results


# ── Aggregated Search Engine ──────────────────────────────────────────


class SearchEngine:
    """Aggregates multiple providers, deduplicates, and caches results.

    Usage:
        engine = SearchEngine()
        results = await engine.search("quantum computing", content_types=["academic", "web"])
    """

    def __init__(
        self,
        providers: list[SearchProvider] | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        self._providers = providers or [
            # Free tier — no API keys needed
            WikipediaProvider(),
            DuckDuckGoProvider(),
            ArxivProvider(),
            CrossRefProvider(),
            OpenLibraryProvider(),
            USGovernmentProvider(),
            LibraryOfCongressProvider(),
            # Optional: Add these if you have API keys (free tier available but limited)
            # SemanticScholarProvider(api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
            # OpenAlexProvider(api_key=os.getenv("OPENALEX_API_KEY")),
            # GNewsProvider(api_key=os.getenv("GNEWS_API_KEY")),
            # NYTimesProvider(api_key=os.getenv("NYT_API_KEY")),
            # FREDProvider(api_key=os.getenv("FRED_API_KEY")),
            # GitHubSearchProvider(token=os.getenv("GITHUB_TOKEN")),
        ]
        self._cache = cache or DiskCache()
        self._provider_cooldowns: dict[str, float] = {}

    async def __aenter__(self) -> SearchEngine:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def _provider_available(self, provider: SearchProvider) -> bool:
        cooldown_until = self._provider_cooldowns.get(provider.name)
        return cooldown_until is None or time.monotonic() >= cooldown_until

    def _cooldown_seconds(self, status_code: int, response: httpx.Response) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass

        if status_code == 403:
            return 30 * 60
        if status_code == 429:
            return 10 * 60
        if status_code in {502, 503, 504}:
            return 60.0
        return 5 * 60

    def _mark_provider_failure(self, provider: SearchProvider, exc: httpx.HTTPStatusError) -> None:
        cooldown_seconds = self._cooldown_seconds(exc.response.status_code, exc.response)
        self._provider_cooldowns[provider.name] = time.monotonic() + cooldown_seconds
        logger.info(
            "%s returned %s; pausing retries for %.0f seconds",
            provider.name,
            exc.response.status_code,
            cooldown_seconds,
        )

    async def search(
        self,
        query: str,
        max_results_per_provider: int = 5,
        content_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search across all providers concurrently."""
        providers = [provider for provider in self._providers if self._provider_available(provider)]
        if content_types:
            providers = [p for p in providers if p.content_type in content_types]

        tasks = [
            self._search_single(provider, query, max_results_per_provider) for provider in providers
        ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[SearchResult] = []
        for batch in all_results:
            if isinstance(batch, BaseException):
                logger.warning("Provider search failed: %s", batch)
                continue
            results.extend(batch)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        deduped: list[SearchResult] = []
        for r in results:
            normalized_url = r.url.rstrip("/").lower()
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                deduped.append(r)

        return deduped

    async def _search_single(
        self, provider: SearchProvider, query: str, max_results: int
    ) -> list[SearchResult]:
        cached = self._cache.get(provider.name, query)
        if cached is not None:
            return [SearchResult(**r) for r in cached]

        try:
            results = await provider.search(query, max_results)
            self._cache.set(
                provider.name,
                query,
                [
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                        "source_provider": r.source_provider,
                        "domain": r.domain,
                        "published_date": r.published_date,
                        "authors": r.authors,
                        "doi": r.doi,
                        "citation_count": r.citation_count,
                        "content_type": r.content_type,
                        "raw_metadata": r.raw_metadata,
                    }
                    for r in results
                ],
            )
            return results
        except httpx.HTTPStatusError as e:
            if e.response.status_code in {403, 429, 502, 503, 504}:
                self._mark_provider_failure(provider, e)
            logger.warning("%s failed for '%s': %s", provider.name, query, e)
            return []
        except Exception as e:
            logger.warning("%s failed for '%s': %s", provider.name, query, e)
            return []

    async def close(self) -> None:
        for provider in self._providers:
            await provider.close()
