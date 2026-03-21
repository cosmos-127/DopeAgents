"""MCP server factory and integration."""

from typing import Any, Optional


def register_agent_as_mcp_tool(agent: Any, server: Optional[Any] = None) -> Any:
    """Register an agent as an MCP tool on a server.
    
    Args:
        agent: Agent instance to register
        server: FastMCP server instance. If None, creates a new one.
        
    Returns:
        The MCP tool registration or server.
        
    Stub implementation - real implementation requires fastmcp.
    """
    raise ImportError("MCP support requires: pip install dopeagents[mcp]")


def create_single_agent_mcp_server(agent: Any) -> Any:
    """Create a standalone MCP server exposing a single agent as a tool.
    
    Args:
        agent: Agent instance to expose
        
    Returns:
        FastMCP server ready to run
        
    Stub implementation - real implementation requires fastmcp.
    """
    raise ImportError("MCP support requires: pip install dopeagents[mcp]")
