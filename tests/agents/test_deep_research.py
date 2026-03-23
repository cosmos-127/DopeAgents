"""Tests for the enhanced research agent deep_research infrastructure."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dopeagents.agent_utils.content_extractor import ContentExtractor
from dopeagents.agent_utils.credibility import score_credibility
from dopeagents.agent_utils.fact_checker import FactChecker, FactCheckResult
from dopeagents.agent_utils.search_providers import (
    DiskCache,
    RateLimiter,
    SearchEngine,
    SearchResult,
    WikipediaProvider,
)

# ── Credibility Tests ──────────────────────────────────────────────────


class TestCredibilityScoring:
    def test_high_authority_domain(self) -> None:
        score = score_credibility(
            url="https://www.nature.com/articles/s41586-023-01234",
            published_date="2025-06-15",
            citation_count=150,
            content_type="academic",
            has_author=True,
            word_count=5000,
        )
        assert score.overall >= 0.85
        assert score.domain_authority >= 0.90

    def test_low_authority_domain(self) -> None:
        score = score_credibility(
            url="https://medium.com/@random-user/my-thoughts",
            published_date="2023-01-01",
            content_type="web",
            has_author=False,
            word_count=200,
        )
        assert score.overall < 0.55
        assert score.domain_authority <= 0.40

    def test_edu_domain_boost(self) -> None:
        score = score_credibility(
            url="https://cs.stanford.edu/research/paper.html",
            content_type="academic",
        )
        assert score.domain_authority >= 0.80

    def test_old_paper_recency_penalty(self) -> None:
        score = score_credibility(
            url="https://example.com/paper",
            published_date="2005-01-01",
        )
        assert score.recency_score <= 0.4

    def test_highly_cited_bonus(self) -> None:
        score = score_credibility(
            url="https://example.com/paper",
            citation_count=500,
        )
        assert score.citation_score >= 0.9

    def test_wikipedia_moderate_credibility(self) -> None:
        score = score_credibility(
            url="https://en.wikipedia.org/wiki/Machine_learning",
            content_type="encyclopedia",
        )
        assert 0.55 <= score.overall <= 0.80

    def test_gov_domain_boost(self) -> None:
        score = score_credibility(
            url="https://data.census.gov/results",
            content_type="web",
        )
        assert score.domain_authority >= 0.85

    def test_unknown_domain_gets_default(self) -> None:
        score = score_credibility(
            url="https://randomsite123.xyz/article",
            content_type="web",
        )
        assert score.domain_authority == 0.50

    def test_recent_publication_bonus(self) -> None:
        score = score_credibility(
            url="https://example.com/paper",
            published_date="2026-01-01",
        )
        assert score.recency_score == 1.0

    def test_no_date_gets_default_recency(self) -> None:
        score = score_credibility(url="https://example.com/paper")
        assert score.recency_score == 0.5


# ── Search Provider Tests ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestWikipediaProviderMocked:
    async def test_search_with_mock(self) -> None:
        provider = WikipediaProvider()
        mock_response = httpx.Response(
            200,
            json={
                "query": {
                    "search": [
                        {
                            "title": "Machine learning",
                            "snippet": "ML is a subset of <span>AI</span>...",
                            "timestamp": "2024-01-15T00:00:00Z",
                        }
                    ]
                }
            },
            request=httpx.Request("GET", "https://en.wikipedia.org/w/api.php"),
        )
        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = client

            results = await provider.search("machine learning")
            assert len(results) == 1
            assert results[0].title == "Machine learning"
            assert results[0].source_provider == "wikipedia"
            assert results[0].domain == "en.wikipedia.org"
            # HTML tags should be stripped
            assert "<span>" not in results[0].snippet

    async def test_empty_results(self) -> None:
        provider = WikipediaProvider()
        mock_response = httpx.Response(
            200,
            json={"query": {"search": []}},
            request=httpx.Request("GET", "https://en.wikipedia.org/w/api.php"),
        )
        with patch.object(provider, "_get_client") as mock_client:
            client = AsyncMock()
            client.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = client

            results = await provider.search("xyznonexistent")
            assert len(results) == 0


# ── Search Engine Tests ───────────────────────────────────────────────


@pytest.mark.asyncio
class TestSearchEngine:
    async def test_aggregated_search_deduplicates(self) -> None:
        result_1 = SearchResult(
            title="Result 1",
            url="https://example.com/1",
            snippet="First result",
            source_provider="provider_a",
        )
        result_2 = SearchResult(
            title="Result 2",
            url="https://example.com/2",
            snippet="Second result",
            source_provider="provider_b",
        )
        # Duplicate URL
        result_dup = SearchResult(
            title="Result 1 duplicate",
            url="https://example.com/1",
            snippet="Duplicate",
            source_provider="provider_b",
        )

        provider_a = MagicMock()
        provider_a.content_type = "web"
        provider_a.search = AsyncMock(return_value=[result_1])

        provider_b = MagicMock()
        provider_b.content_type = "web"
        provider_b.search = AsyncMock(return_value=[result_2, result_dup])

        engine = SearchEngine(providers=[provider_a, provider_b])
        results = await engine.search("test query")

        assert len(results) == 2
        urls = {r.url for r in results}
        assert "https://example.com/1" in urls
        assert "https://example.com/2" in urls

    async def test_filters_by_content_type(self) -> None:
        provider_web = MagicMock()
        provider_web.content_type = "web"
        provider_web.search = AsyncMock(return_value=[])

        provider_academic = MagicMock()
        provider_academic.content_type = "academic"
        provider_academic.search = AsyncMock(
            return_value=[
                SearchResult(
                    title="Paper",
                    url="https://arxiv.org/paper1",
                    snippet="Academic paper",
                    source_provider="arxiv",
                )
            ]
        )

        engine = SearchEngine(providers=[provider_web, provider_academic])
        results = await engine.search("test", content_types=["academic"])

        # Only academic provider should be searched
        provider_web.search.assert_not_awaited()
        provider_academic.search.assert_awaited_once()
        assert len(results) == 1

    async def test_handles_provider_failure(self) -> None:
        provider_ok = MagicMock()
        provider_ok.content_type = "web"
        provider_ok.search = AsyncMock(
            return_value=[
                SearchResult(
                    title="Good",
                    url="https://example.com/good",
                    snippet="Works",
                    source_provider="ok",
                )
            ]
        )

        provider_bad = MagicMock()
        provider_bad.content_type = "web"
        provider_bad.name = "bad_provider"
        provider_bad.search = AsyncMock(side_effect=Exception("API down"))

        engine = SearchEngine(providers=[provider_ok, provider_bad])
        results = await engine.search("test")

        assert len(results) == 1
        assert results[0].title == "Good"

    async def test_skips_provider_after_http_failure(self) -> None:
        provider_ok = MagicMock()
        provider_ok.content_type = "web"
        provider_ok.name = "provider_ok"
        provider_ok.search = AsyncMock(
            return_value=[
                SearchResult(
                    title="Good",
                    url="https://example.com/good",
                    snippet="Works",
                    source_provider="ok",
                )
            ]
        )

        provider_bad = MagicMock()
        provider_bad.content_type = "web"
        provider_bad.name = "bad_provider"
        bad_response = httpx.Response(
            403,
            request=httpx.Request("GET", "https://en.wikipedia.org/w/api.php"),
        )
        provider_bad.search = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Forbidden",
                request=bad_response.request,
                response=bad_response,
            )
        )

        engine = SearchEngine(providers=[provider_ok, provider_bad])

        first_results = await engine.search("test")
        second_results = await engine.search("test")

        assert len(first_results) == 1
        assert len(second_results) == 1
        assert provider_bad.search.await_count == 1
        assert provider_ok.search.await_count == 2


# ── Content Extractor Tests ───────────────────────────────────────────


@pytest.mark.asyncio
class TestContentExtractor:
    async def test_extract_non_http_url(self) -> None:
        extractor = ContentExtractor()
        result = await extractor.extract("ftp://example.com/file")
        assert not result.success
        assert result.extraction_method == "skip"

    async def test_extract_with_mock_html(self) -> None:
        extractor = ContentExtractor()
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <article>
                <p>This is a test article with substantial content about machine learning
                and artificial intelligence research. trafilatura requires >100 chars
                of extracted text; BeautifulSoup falls back for shorter content.</p>
            </article>
        </body>
        </html>
        """
        mock_response = httpx.Response(
            200,
            text=html,
            request=httpx.Request("GET", "https://example.com/article"),
        )
        with patch.object(extractor, "_get_client") as mock_client:
            client = AsyncMock()
            client.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = client

            result = await extractor.extract("https://example.com/article")
            assert result.success
            assert result.word_count > 0
            assert "machine learning" in result.full_text.lower()

    async def test_extract_handles_fetch_failure(self) -> None:
        extractor = ContentExtractor()
        with patch.object(extractor, "_get_client") as mock_client:
            client = AsyncMock()
            client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client.return_value = client

            result = await extractor.extract("https://example.com/slow")
            assert not result.success
            assert result.extraction_method == "fetch_failed"

    async def test_extract_batch(self) -> None:
        extractor = ContentExtractor()
        html = "<html><body><p>Content here with enough words to pass threshold easily</p></body></html>"
        mock_response = httpx.Response(
            200,
            text=html,
            request=httpx.Request("GET", "https://example.com/batch"),
        )
        with patch.object(extractor, "_get_client") as mock_client:
            client = AsyncMock()
            client.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = client

            results = await extractor.extract_batch(
                ["https://example.com/1", "https://example.com/2"],
                max_concurrent=2,
            )
            assert len(results) == 2


# ── Disk Cache Tests ──────────────────────────────────────────────────


class TestDiskCache:
    def test_set_and_get(self, tmp_path: Path) -> None:
        cache = DiskCache(cache_dir=str(tmp_path / ".cache"))
        cache.set("test_provider", "test query", [{"title": "Result 1"}])

        result = cache.get("test_provider", "test query")
        assert result is not None
        assert len(result) == 1
        assert result[0]["title"] == "Result 1"

    def test_cache_miss(self, tmp_path: Path) -> None:
        cache = DiskCache(cache_dir=str(tmp_path / ".cache"))
        result = cache.get("test_provider", "nonexistent query")
        assert result is None


# ── Rate Limiter Tests ────────────────────────────────────────────────


@pytest.mark.asyncio
class TestRateLimiter:
    async def test_rate_limiting(self) -> None:
        limiter = RateLimiter(calls_per_second=10.0)  # 100ms between calls
        start = time.monotonic()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.08  # Should have waited ~100ms (margin for CI)


# ── SearchResult Tests ────────────────────────────────────────────────


class TestSearchResult:
    def test_domain_auto_populated(self) -> None:
        r = SearchResult(
            title="Test",
            url="https://www.nature.com/articles/123",
            snippet="Test snippet",
            source_provider="test",
        )
        assert r.domain == "www.nature.com"

    def test_empty_url_no_domain(self) -> None:
        r = SearchResult(
            title="Test",
            url="",
            snippet="Test",
            source_provider="test",
        )
        assert r.domain == ""


# ── Fact Checker Tests ───────────────────────────────────────────────


@pytest.mark.asyncio
class TestFactChecker:
    async def test_no_api_key_skips_google(self) -> None:
        checker = FactChecker()  # no google_api_key
        with (
            patch.object(
                checker, "_wikipedia_verify", new=AsyncMock(return_value={"found": False})
            ),
            patch.object(checker, "_wikidata_verify", new=AsyncMock(return_value={"found": False})),
        ):
            result = await checker.check_claim("The sky is blue")
        assert "Google Fact Check" not in result.sources_checked
        assert result.verified is None
        assert result.confidence == 0.0

    async def test_google_true_rating_sets_verified(self) -> None:
        checker = FactChecker(google_api_key="fake_key")
        with (
            patch.object(
                checker,
                "_google_fact_check",
                new=AsyncMock(
                    return_value=[{"textualRating": "True", "publisher": {"name": "Snopes"}}]
                ),
            ),
            patch.object(
                checker, "_wikipedia_verify", new=AsyncMock(return_value={"found": False})
            ),
            patch.object(checker, "_wikidata_verify", new=AsyncMock(return_value={"found": False})),
        ):
            result = await checker.check_claim("Vaccines are effective")
        assert result.verified is True
        assert result.confidence >= 0.9
        assert "Google Fact Check" in result.sources_checked
        assert any("true" in e.lower() for e in result.supporting_evidence)

    async def test_google_false_rating_sets_contradicted(self) -> None:
        checker = FactChecker(google_api_key="fake_key")
        with (
            patch.object(
                checker,
                "_google_fact_check",
                new=AsyncMock(
                    return_value=[{"textualRating": "False", "publisher": {"name": "PolitiFact"}}]
                ),
            ),
            patch.object(
                checker, "_wikipedia_verify", new=AsyncMock(return_value={"found": False})
            ),
            patch.object(checker, "_wikidata_verify", new=AsyncMock(return_value={"found": False})),
        ):
            result = await checker.check_claim("The earth is flat")
        assert result.verified is False
        assert result.confidence >= 0.9
        assert len(result.contradicting_evidence) > 0

    async def test_wikipedia_found_contributes_support(self) -> None:
        checker = FactChecker()
        with (
            patch.object(
                checker,
                "_wikipedia_verify",
                new=AsyncMock(
                    return_value={
                        "found": True,
                        "excerpt": "Machine learning is a subset of AI",
                        "title": "Machine learning",
                    }
                ),
            ),
            patch.object(checker, "_wikidata_verify", new=AsyncMock(return_value={"found": False})),
        ):
            result = await checker.check_claim("machine learning is AI")
        assert "Wikipedia" in result.sources_checked
        assert result.verified is True
        assert result.confidence >= 0.6

    async def test_wikidata_found_boosts_confidence(self) -> None:
        checker = FactChecker()
        with (
            patch.object(
                checker, "_wikipedia_verify", new=AsyncMock(return_value={"found": False})
            ),
            patch.object(
                checker,
                "_wikidata_verify",
                new=AsyncMock(
                    return_value={
                        "found": True,
                        "entity_id": "Q11660",
                        "value": "field of study",
                        "label": "machine learning",
                    }
                ),
            ),
        ):
            result = await checker.check_claim("machine learning")
        assert "Wikidata" in result.sources_checked
        assert result.confidence >= 0.8
        assert result.verified is True

    async def test_all_sources_fail_returns_unverifiable(self) -> None:
        checker = FactChecker()
        with (
            patch.object(
                checker,
                "_wikipedia_verify",
                new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
            ),
            patch.object(
                checker, "_wikidata_verify", new=AsyncMock(side_effect=Exception("network error"))
            ),
        ):
            result = await checker.check_claim("some unverifiable claim")
        assert result.verified is None
        assert result.confidence == 0.0
        assert result.sources_checked == []
        assert result.method == "multi-source"

    async def test_check_batch_returns_all_results(self) -> None:
        checker = FactChecker()
        with (
            patch.object(
                checker, "_wikipedia_verify", new=AsyncMock(return_value={"found": False})
            ),
            patch.object(checker, "_wikidata_verify", new=AsyncMock(return_value={"found": False})),
        ):
            results = await checker.check_batch(["claim one", "claim two", "claim three"])
        assert len(results) == 3
        assert all(isinstance(r, FactCheckResult) for r in results)
