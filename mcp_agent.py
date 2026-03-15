"""
MCP Agent — Tutorial

Connects to mcp_db_server.py at runtime and uses its tools to answer
database questions. Answers the same questions as db_agent.py, but the
tools now live in a separate server process.

──────────────────────────────────────────────────────────────────────────────
WHAT'S DIFFERENT FROM db_agent.py
──────────────────────────────────────────────────────────────────────────────
  db_agent.py:
    - TOOLS list is hardcoded JSON schemas in the agent file
    - Agent calls Python functions directly (execute_tool)

  mcp_agent.py (this file):
    - No tool definitions here at all
    - On startup, the agent connects to the server and calls tools/list
      to discover what's available dynamically
    - Tool calls go through the MCP protocol: session.call_tool(name, args)
    - The server could be written in any language, running anywhere

──────────────────────────────────────────────────────────────────────────────
THE STDIO TRANSPORT
──────────────────────────────────────────────────────────────────────────────
stdio_client() launches mcp_db_server.py as a subprocess and connects to it
over stdin/stdout. The protocol is JSON-RPC 2.0.

You can swap this for an HTTP/SSE transport to connect to a remote server
with no changes to the agent loop.

──────────────────────────────────────────────────────────────────────────────
Run it:
  python mcp_agent.py "Which director has the most movies in the database?"
  python mcp_agent.py "What are the top 5 stocks by market cap?"
──────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import json
import sys

import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "claude-sonnet-4-6"

SYSTEM = """\
You are a data research assistant with access to a SQLite database.
To answer questions, you should:
  1. Call list_tables_tool to see what's available.
  2. Call describe_table_tool on the relevant table(s) to learn column names.
  3. Write and run a SELECT query with run_query_tool.
  4. If run_query_tool returns a result starting with "Error:", fix the SQL and retry.
  5. Summarize the query results in plain English as your final answer.

Always base your answers on actual query results — do not guess."""


def mcp_tool_to_anthropic(tool) -> dict:
    """
    Convert an MCP Tool object to the dict format Anthropic's API expects.

    MCP tools have:  tool.name, tool.description, tool.inputSchema
    Anthropic wants: {"name": ..., "description": ..., "input_schema": ...}

    The schema contents are identical (both are JSON Schema objects) —
    only the key name differs.
    """
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema,
    }


def extract_tool_result(call_result) -> str:
    """
    Pull a plain string out of a CallToolResult.

    MCP returns a list of content blocks (TextContent, ImageContent, etc.).
    For these DB tools they'll always be TextContent, so we join the text fields.
    """
    parts = []
    for block in call_result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts) if parts else "(empty result)"


async def run_agent(user_message: str) -> str:
    client = anthropic.Anthropic()

    # ── Connect to the MCP server ──────────────────────────────────────────────
    # StdioServerParameters tells the client how to launch the server process.
    # The client starts it as a subprocess and speaks to it over stdio.
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp_db_server.py"],
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:

            # Handshake — must be called before anything else
            await session.initialize()

            # ── Discover tools at runtime ──────────────────────────────────────
            # This is the key difference from db_agent.py: we don't hardcode
            # the tool schemas. We ask the server what it provides.
            tools_result = await session.list_tools()
            anthropic_tools = [mcp_tool_to_anthropic(t) for t in tools_result.tools]

            print(f"\nQuestion: {user_message}")
            print(f"Tools discovered from server: {[t['name'] for t in anthropic_tools]}\n")

            messages = [{"role": "user", "content": user_message}]

            # ── Agentic loop (identical structure to db_agent.py) ─────────────
            iteration = 0
            while True:
                iteration += 1
                print(f"[API call #{iteration}]")

                response = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    system=SYSTEM,
                    tools=anthropic_tools,
                    messages=messages,
                )
                print(f"  stop_reason : {response.stop_reason}")
                print(f"  input_tokens: {response.usage.input_tokens}, output_tokens: {response.usage.output_tokens}")

                if response.stop_reason == "end_turn":
                    final_text = next(
                        (b.text for b in response.content if b.type == "text"), ""
                    )
                    print(f"\nAnswer: {final_text}")
                    return final_text

                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})

                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            print(f"  → {block.name}({json.dumps(block.input)})")

                            # Call the tool via MCP — goes over the stdio transport
                            # to the server process, which runs the Python function
                            # and sends the result back as a JSON-RPC response.
                            call_result = await session.call_tool(block.name, block.input)
                            result_str = extract_tool_result(call_result)

                            is_error = call_result.isError or result_str.startswith("Error:")
                            preview = result_str[:300] + "…" if len(result_str) > 300 else result_str
                            prefix = "ERROR" if is_error else "result"
                            print(f"    ↳ [{prefix}] {preview}")

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_str,
                                **({"is_error": True} if is_error else {}),
                            })

                    messages.append({"role": "user", "content": tool_results})

                else:
                    print(f"  Unexpected stop_reason: {response.stop_reason!r}. Stopping.")
                    break

    return ""


if __name__ == "__main__":
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What are the top 3 highest-rated movies, and who directed them?"
    )
    asyncio.run(run_agent(query))
