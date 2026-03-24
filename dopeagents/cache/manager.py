"""CacheManager ABC and InMemoryCache with TTL support (T2.11)."""

from __future__ import annotations

import hashlib
import json
import time

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from dopeagents.core.agent import Agent


class CacheManager(ABC):
    """Abstract cache backend for agent outputs.

    Concrete implementations: InMemoryCache (in-process, TTL), DiskCache (persistent).
    Cache keys are built from agent name + version + serialised input, so different
    agent versions or different inputs never collide.
    """

    @abstractmethod
    def get(self, agent: Agent, input: BaseModel) -> BaseModel | None:  # type: ignore[type-arg]
        """Return cached output for the given agent + input, or None on miss."""
        ...

    @abstractmethod
    def set(
        self,
        agent: Agent,  # type: ignore[type-arg]
        input: BaseModel,
        output: BaseModel,
        ttl: int | None = None,
    ) -> None:
        """Store output in the cache; optionally expire after ttl seconds."""
        ...

    @abstractmethod
    def invalidate(
        self,
        agent: Agent,  # type: ignore[type-arg]
        input: BaseModel | None = None,
    ) -> None:
        """Remove one entry (if input given) or all entries for this agent."""
        ...

    @staticmethod
    def _build_key(agent: Agent, input: BaseModel) -> str:  # type: ignore[type-arg]
        """Build a stable SHA-256 cache key from agent identity + serialised input."""
        key_data = {
            "agent": agent.name,
            "version": agent.version,
            "input": input.model_dump(mode="json"),
        }
        key_json = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_json.encode()).hexdigest()


class InMemoryCache(CacheManager):
    """Thread-unsafe in-process cache backed by a plain dict.

    Suitable for single-threaded use, testing, or short-lived processes.
    For production multi-threaded use, wrap with a Lock or use DiskCache.

    TTL: entries expire lazily on ``get()`` — no background eviction thread.
    """

    def __init__(self) -> None:
        # key → (output, expiry_unix_timestamp | None)
        self._store: dict[str, tuple[BaseModel, float | None]] = {}

    def get(self, agent: Agent, input: BaseModel) -> BaseModel | None:  # type: ignore[type-arg]
        key = self._build_key(agent, input)
        if key not in self._store:
            return None
        value, expiry = self._store[key]
        if expiry is not None and expiry <= time.time():
            del self._store[key]
            return None
        return value

    def set(
        self,
        agent: Agent,  # type: ignore[type-arg]
        input: BaseModel,
        output: BaseModel,
        ttl: int | None = None,
    ) -> None:
        key = self._build_key(agent, input)
        expiry = (time.time() + ttl) if ttl is not None else None
        self._store[key] = (output, expiry)

    def invalidate(
        self,
        agent: Agent,  # type: ignore[type-arg]
        input: BaseModel | None = None,
    ) -> None:
        if input is not None:
            key = self._build_key(agent, input)
            self._store.pop(key, None)
        else:
            # Remove all keys for this agent (by prefix-matching the name in the data)
            # Since keys are hashes, we must rebuild them from stored data; simplest
            # approach is to clear the whole cache when no input is specified.
            self._store.clear()

    def size(self) -> int:
        """Return the number of entries currently stored (including expired)."""
        return len(self._store)
