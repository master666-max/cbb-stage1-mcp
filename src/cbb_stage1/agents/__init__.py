"""Conversation-scoped subagents over external inference endpoints."""

from .subagent import (
    ROLE_PROMPTS,
    EndpointSubagent,
    SubagentLLM,
    SubagentMessage,
    SubagentPool,
    SubagentResult,
    SubagentTask,
    build_subagent_pool,
)
from .mcp_model_client import (
    MCPError,
    MCPModelClient,
    MCPStdioTransport,
    build_mcp_model_client,
)

__all__ = [
    "ROLE_PROMPTS",
    "EndpointSubagent",
    "MCPError",
    "MCPModelClient",
    "MCPStdioTransport",
    "SubagentLLM",
    "SubagentMessage",
    "SubagentPool",
    "SubagentResult",
    "SubagentTask",
    "build_mcp_model_client",
    "build_subagent_pool",
]
