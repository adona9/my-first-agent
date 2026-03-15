"""
MCP Server — Research DB

Exposes the same three database tools from db_agent.py as an MCP server.

──────────────────────────────────────────────────────────────────────────────
WHAT IS MCP?
──────────────────────────────────────────────────────────────────────────────
MCP (Model Context Protocol) is an open standard for connecting AI models to
external tools and data sources. It decouples the tool *implementation* from
the agent that uses it.

Before MCP (what you built in db_agent.py):
  - Tools are defined as JSON schemas inside the agent file
  - The agent calls the Python functions directly
  - Tools are tightly coupled to one agent

With MCP:
  - Tools live in a standalone *server* process (this file)
  - Any MCP-compatible *client* can discover and call them at runtime
  - One server can serve Claude, GPT-4, Cursor, your IDE — anything

The transport here is *stdio*: the client starts this script as a subprocess
and communicates over stdin/stdout using JSON-RPC.

──────────────────────────────────────────────────────────────────────────────
Run standalone to verify the server starts:
  uv run python mcp_db_server.py

(It will block waiting for a client — Ctrl-C to stop.)
──────────────────────────────────────────────────────────────────────────────
"""

import logging

from mcp.server.fastmcp import FastMCP

from db_tools import describe_table, list_tables, run_query

# Suppress the "Processing request of type ..." lines the MCP framework
# emits to stderr — they're noise when the server is embedded in an agent.
logging.disable(logging.WARNING)

# FastMCP handles the protocol boilerplate: JSON-RPC framing, tools/list,
# tools/call, error responses. You just decorate functions.
mcp = FastMCP("research-db")


# The @mcp.tool() decorator does the same job as the TOOLS list in db_agent.py:
#   - function name  → tool name
#   - docstring      → tool description (the model reads this)
#   - type hints     → input schema (FastMCP generates JSON Schema for you)
#
# This is identical in spirit to LangChain's @tool decorator.

@mcp.tool()
def list_tables_tool() -> str:
    """List all tables in the research database along with their row counts.
    Always call this first when you don't know what data is available."""
    return list_tables()


@mcp.tool()
def describe_table_tool(table_name: str) -> str:
    """Show the column names, types, and a few sample rows for a table.
    Call this before writing a query so you know the exact column names.

    Args:
        table_name: Name of the table to describe.
    """
    return describe_table(table_name)


@mcp.tool()
def run_query_tool(sql: str) -> str:
    """Execute a SQL SELECT query against the research database and get results
    as a JSON array. If the query fails the result starts with 'Error:' — fix
    the SQL and retry. Only SELECT queries are allowed.

    Args:
        sql: A valid SQLite SELECT statement.
    """
    return run_query(sql)


if __name__ == "__main__":
    # mcp.run() starts the stdio transport — blocks until the client disconnects
    mcp.run()
