"""Extract and process full text from URLs."""

from __future__ import annotations

import asyncio
import logging
import re

from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Full extracted content from a URL."""

    url: str
    title: str
    full_text: str
    word_count: int
    extraction_method: str
    success: bool
    error: str | None = None


class ContentExtractor:
    """Extracts readable text from web pages.

    Tries in order:
    1. trafilatura (best for articles)
    2. BeautifulSoup fallback
    3. Regex-based raw extraction
    """

    def __init__(self, max_content_length: int = 10_000) -> None:
        self._max_length = max_content_length
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=20.0,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; ResearchAgent/2.0; "
                        "+https://github.com/cosmos-127/dopeagents)"
                    )
                },
                follow_redirects=True,
            )
        return self._client

    async def extract(self, url: str) -> ExtractedContent:
        """Extract readable content from a URL."""
        if not url.startswith(("http://", "https://")):
            return ExtractedContent(
                url=url,
                title="",
                full_text="",
                word_count=0,
                extraction_method="skip",
                success=False,
                error="Non-HTTP URL",
            )

        try:
            client = await self._get_client()
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            return ExtractedContent(
                url=url,
                title="",
                full_text="",
                word_count=0,
                extraction_method="fetch_failed",
                success=False,
                error=str(e),
            )

        # Method 1: trafilatura
        result = self._try_trafilatura(html, url)
        if result:
            return result

        # Method 2: BeautifulSoup
        result = self._try_beautifulsoup(html, url)
        if result:
            return result

        # Method 3: Regex-based raw extraction
        return self._try_regex(html, url)

    def _try_trafilatura(self, html: str, url: str) -> ExtractedContent | None:
        try:
            import trafilatura

            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            if extracted and len(extracted) > 100:
                text = extracted[: self._max_length]
                title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
                title_str = title_match.group(1).strip() if title_match else ""
                return ExtractedContent(
                    url=url,
                    title=title_str,
                    full_text=text,
                    word_count=len(text.split()),
                    extraction_method="trafilatura",
                    success=True,
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug("trafilatura failed for %s: %s", url, e)
        return None

    def _try_beautifulsoup(self, html: str, url: str) -> ExtractedContent | None:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            title_str = soup.title.string if soup.title else ""

            content_tag = soup.find("article") or soup.find("main") or soup.find("body")
            if content_tag:
                text = content_tag.get_text(separator="\n", strip=True)
                text = re.sub(r"\n{3,}", "\n\n", text)
                text = text[: self._max_length]
                return ExtractedContent(
                    url=url,
                    title=title_str or "",
                    full_text=text,
                    word_count=len(text.split()),
                    extraction_method="beautifulsoup",
                    success=True,
                )
        except ImportError:
            pass
        except Exception as e:
            logger.debug("BeautifulSoup failed for %s: %s", url, e)
        return None

    def _try_regex(self, html: str, url: str) -> ExtractedContent:
        try:
            title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title_str = title_match.group(1).strip() if title_match else ""

            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            text = text[: self._max_length]

            return ExtractedContent(
                url=url,
                title=title_str,
                full_text=text,
                word_count=len(text.split()),
                extraction_method="regex",
                success=True,
            )
        except Exception as e:
            return ExtractedContent(
                url=url,
                title="",
                full_text="",
                word_count=0,
                extraction_method="failed",
                success=False,
                error=str(e),
            )

    async def extract_batch(
        self, urls: list[str], max_concurrent: int = 5
    ) -> list[ExtractedContent]:
        """Extract content from multiple URLs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited_extract(u: str) -> ExtractedContent:
            async with semaphore:
                return await self.extract(u)

        return await asyncio.gather(*[_limited_extract(u) for u in urls])

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
