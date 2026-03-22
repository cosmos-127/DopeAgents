"""Persistent disk-based cache via diskcache (T2.12).

Requires: pip install dopeagents[cache]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from dopeagents.cache.manager import CacheManager

if TYPE_CHECKING:
    from dopeagents.core.agent import Agent


def _check_installed() -> None:
    try:
        import diskcache  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "diskcache is required for DiskCache.\nInstall with: pip install dopeagents[cache]"
        ) from e


class DiskCache(CacheManager):
    """Persistent cache backed by the diskcache library.

    Survives process restarts and is safe for multi-process use.
    Cache is stored in a directory specified at construction time.

    Args:
        directory: Path to the cache directory (created automatically).
            Defaults to ``.dopeagents_cache`` in the current working directory.
    """

    def __init__(self, directory: str = ".dopeagents_cache") -> None:
        _check_installed()
        import diskcache

        self._cache = diskcache.Cache(directory)

    def get(self, agent: Agent, input: BaseModel) -> BaseModel | None:  # type: ignore[type-arg]
        key = self._build_key(agent, input)
        value = self._cache.get(key)
        if value is None:
            return None
        return value  # type: ignore[no-any-return]

    def set(
        self,
        agent: Agent,  # type: ignore[type-arg]
        input: BaseModel,
        output: BaseModel,
        ttl: int | None = None,
    ) -> None:
        key = self._build_key(agent, input)
        self._cache.set(key, output, expire=ttl)

    def invalidate(
        self,
        agent: Agent,  # type: ignore[type-arg]
        input: BaseModel | None = None,
    ) -> None:
        if input is not None:
            key = self._build_key(agent, input)
            self._cache.delete(key)
        else:
            self._cache.clear()

    def close(self) -> None:
        """Close the underlying diskcache database connection."""
        self._cache.close()
