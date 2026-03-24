"""
Comprehensive test suite for all 18+ search providers.
Tests both free-tier and key-gated providers.
Checks if upstream APIs are working and collects diagnostics.
"""

import asyncio
import time

from datetime import datetime

from dopeagents.agent_utils.search_providers import (
    ArxivProvider,
    CrossRefProvider,
    DuckDuckGoProvider,
    LibraryOfCongressProvider,
    OpenLibraryProvider,
    SearchEngine,
    USGovernmentProvider,
    WikipediaProvider,
)


async def test_provider(provider, test_query: str = "artificial intelligence") -> dict:
    """Test a single provider and return diagnostics."""
    provider_name = provider.name
    start_time = time.time()

    print(f"\n{'=' * 70}")
    print(f"Testing: {provider_name.upper()}")
    print(f"{'=' * 70}")

    try:
        results = await provider.search(test_query, max_results=3)
        elapsed = time.time() - start_time

        print("[OK] Status: WORKING")
        print(f"     Response time: {elapsed:.2f}s")
        print(f"     Results returned: {len(results)}")

        if results:
            for i, r in enumerate(results[:2], 1):
                print(f"\n     Result {i}:")
                print(f"       Title: {r.title[:60]}...")
                print(f"       URL: {r.url[:70]}...")
                print(f"       Domain: {r.domain}")
                print(f"       Content type: {r.content_type}")

        return {
            "name": provider_name,
            "status": "[OK] WORKING",
            "response_time": elapsed,
            "results_count": len(results),
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)[:100]
        print("[FAIL] Status: FAILED")
        print(f"       Error: {error_msg}")
        print(f"       Response time: {elapsed:.2f}s")

        return {
            "name": provider_name,
            "status": "[FAIL] FAILED",
            "response_time": elapsed,
            "results_count": 0,
            "error": error_msg,
        }
    finally:
        await provider.close()


async def main():
    """Run diagnostics on all 18+ providers."""

    print("\n" + "=" * 80)
    print("DOPEGENTS SEARCH PROVIDER HEALTH CHECK")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Free-tier providers (no keys required)
    free_providers = [
        WikipediaProvider(),
        DuckDuckGoProvider(),
        ArxivProvider(),
        CrossRefProvider(),
        OpenLibraryProvider(),
        USGovernmentProvider(),
        LibraryOfCongressProvider(),
    ]

    print(f"\nTesting {len(free_providers)} free-tier providers (no API keys needed)...\n")

    all_providers = [(p.name.upper(), p) for p in free_providers]

    results = []
    for _name, provider in all_providers:
        result = await test_provider(provider)
        results.append(result)
        await asyncio.sleep(0.5)  # Rate limiting between providers

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    working = [r for r in results if "WORKING" in r["status"]]
    failed = [r for r in results if "FAILED" in r["status"]]

    print(f"\nTotal Providers Tested: {len(results)}")
    print(f"[OK] Working: {len(working)}")
    print(f"[FAIL] Failed: {len(failed)}")

    print("\nProvider Status:")
    for r in results:
        status_symbol = "[OK]" if "WORKING" in r["status"] else "[FAIL]"
        print(
            f"  {status_symbol} {r['name'].ljust(20)} | {r['response_time']:.2f}s | {r['results_count']:2d} results"
        )
        if r["error"]:
            print(f"       Error: {r['error'][:70]}...")

    # Test aggregated search engine
    print("\n" + "=" * 80)
    print("AGGREGATED SEARCH ENGINE TEST")
    print("=" * 80)

    engine = SearchEngine()
    try:
        print("Testing aggregated search across all available providers...")
        agg_results = await engine.search("machine learning", max_results_per_provider=2)
        providers_used = {r.source_provider for r in agg_results}
        print(
            f"[OK] Aggregated search OK: {len(agg_results)} total results from {len(providers_used)} providers"
        )
        print(f"     Providers used: {', '.join(sorted(providers_used))}")

        # Show content type distribution
        content_types = {}
        for r in agg_results:
            ct = r.content_type
            content_types[ct] = content_types.get(ct, 0) + 1
        print(
            f"     Content types: {', '.join(f'{k}({v})' for k, v in sorted(content_types.items()))}"
        )
    except Exception as e:
        print(f"[FAIL] Aggregated search failed: {e}")
    finally:
        await engine.close()

    # Cleanup
    for _, provider in all_providers:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(main())
