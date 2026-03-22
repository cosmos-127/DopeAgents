"""LangChain adapter — wraps agents as LangChain runnables."""

from typing import Any


def agent_to_langchain_runnable(agent: Any) -> Any:
    """Convert agent to LangChain Runnable.

    Stub implementation. Real implementation requires langchain-core.
    """
    raise NotImplementedError("LangChain adapter requires langchain-core")
