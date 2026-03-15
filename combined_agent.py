"""
Combined Agent — LangChain + Persistent Memory + MCP

Brings together every concept from the tutorial series into a single agent.

──────────────────────────────────────────────────────────────────────────────
WHAT'S COMBINED HERE
──────────────────────────────────────────────────────────────────────────────
  LangChain  (from langchain_db_agent.py)
    - ChatAnthropic model wrapper
    - @tool decorator for memory tools
    - create_agent loop (no hand-written while loop)
    - graph.astream() for step-by-step observability (async, because the MCP
      tools are async — they talk to a subprocess over the stdio transport)

  Persistent memory  (from memory_agent.py)
    - Memories loaded from disk and injected into the system prompt each run
    - save / search / delete memory tools so the agent manages its own memory
    - Corrections work: delete the old fact, save the new one

  MCP server  (from mcp_agent.py)
    - DB tools live in mcp_db_server.py, a separate process
    - Agent discovers them at startup via session.list_tools()
    - Each MCP tool is wrapped as a LangChain StructuredTool so create_agent
      can use them alongside the in-process memory tools

──────────────────────────────────────────────────────────────────────────────
THE INTEGRATION CHALLENGE
──────────────────────────────────────────────────────────────────────────────
The three systems each use a different tool abstraction:

  Raw SDK    → plain dicts:         {"name": ..., "input_schema": ...}
  LangChain  → BaseTool instances:  @tool or StructuredTool.from_function()
  MCP        → mcp.types.Tool:      tool.name, tool.description, tool.inputSchema

To combine them, we convert MCP tools into LangChain StructuredTools:
  1. Build a Pydantic model from the MCP tool's JSON Schema (for type-safe args)
  2. Wrap an async call to session.call_tool() as the tool's implementation
  3. Pass the resulting StructuredTool to create_agent alongside the @tool memory tools

The MCP session must stay open for the entire agent run (tool calls happen
mid-stream), so the whole agent invocation lives inside the async with block.

──────────────────────────────────────────────────────────────────────────────
Run it:
  python combined_agent.py "My name is Alessandro. What tables do you have access to?"
  python combined_agent.py "What do you know about me? Also, who directed the most movies?"
  python combined_agent.py "What are the top 3 highest-rated movies?"
──────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import sys
from typing import Optional

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import create_model

from memory_store import delete_memory, format_for_prompt, save_memory, search_memories

MODEL = "claude-sonnet-4-6"

# JSON Schema primitive types → Python types (for building Pydantic models)
_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


# ── Memory tools (LangChain @tool) ─────────────────────────────────────────────
# These are defined at module level — they're pure Python functions that don't
# depend on any external session or connection.

@tool
def save_memory_tool(content: str) -> str:
    """Persist a fact to long-term memory. Call this when the user shares
    something worth remembering across conversations: name, preferences,
    goals, past decisions. One clear fact per call."""
    return save_memory(content)


@tool
def search_memory_tool(query: str) -> str:
    """Search existing memories by keyword. Use before saving to avoid
    duplicates, or when the user asks about something mentioned before."""
    return search_memories(query)


@tool
def delete_memory_tool(memory_id: str) -> str:
    """Delete a memory by its id (the 6-char code in brackets). Use this
    when the user corrects something — delete the old fact, then save the
    new one."""
    return delete_memory(memory_id)


MEMORY_TOOLS = [save_memory_tool, search_memory_tool, delete_memory_tool]


# ── MCP → LangChain tool conversion ────────────────────────────────────────────

def _mcp_tool_to_langchain(session: ClientSession, mcp_tool) -> StructuredTool:
    """
    Wrap an MCP tool as a LangChain StructuredTool.

    The three steps:
      1. Build a Pydantic model from the MCP tool's JSON Schema. LangChain uses
         this to validate arguments and generate the schema Claude sees.
      2. Write an async function that calls session.call_tool() and extracts text.
      3. Bundle into a StructuredTool via from_function(coroutine=...).

    Using a closure over `session` and `mcp_tool.name` means the tool stays
    bound to the live session for the duration of the agent run.
    """
    properties = mcp_tool.inputSchema.get("properties", {})
    required = set(mcp_tool.inputSchema.get("required", []))

    fields: dict = {}
    for field_name, schema in properties.items():
        py_type = _JSON_TYPE_MAP.get(schema.get("type", "string"), str)
        if field_name in required:
            fields[field_name] = (py_type, ...)
        else:
            fields[field_name] = (Optional[py_type], None)

    # Always create the model — even for no-arg tools (creates an empty model)
    args_model = create_model(f"{mcp_tool.name}_args", **fields)

    tool_name = mcp_tool.name  # capture for the closure

    async def call(**kwargs) -> str:
        result = await session.call_tool(tool_name, kwargs if kwargs else None)
        parts = [block.text for block in result.content if hasattr(block, "text")]
        return "\n".join(parts) if parts else "(empty result)"

    return StructuredTool.from_function(
        coroutine=call,
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        args_schema=args_model,
    )


# ── System prompt ───────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """
    Assembled fresh on every run so it reflects the current memory file.

    The memory block (if any) is appended after the base instructions,
    giving Claude knowledge of past conversations from turn 1 without
    needing to call a tool to load them.
    """
    base = """\
You are a helpful assistant with two capabilities:

1. Research database — use the DB tools to answer questions about movies,
   countries, and stocks. Always call list_tables_tool first, then
   describe_table_tool, then run_query_tool.

2. Long-term memory — use the memory tools to remember things across
   conversations. Save facts the user tells you about themselves. When
   correcting a memory: delete the old one first, then save the new one."""

    memory_block = format_for_prompt()
    return base + "\n\n" + memory_block if memory_block else base


# ── Agent ───────────────────────────────────────────────────────────────────────

async def run_agent(user_message: str) -> str:
    server_params = StdioServerParameters(
        command="uv", args=["run", "python", "mcp_db_server.py"]
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Discover DB tools from the MCP server and wrap as LangChain tools
            tools_result = await session.list_tools()
            db_tools = [_mcp_tool_to_langchain(session, t) for t in tools_result.tools]

            all_tools = db_tools + MEMORY_TOOLS

            # Print startup context
            print(f"\nUser: {user_message}")
            memory_block = format_for_prompt()
            if memory_block:
                print(f"\n[Memory loaded]\n{memory_block}")
            else:
                print("\n[No memories yet]")
            db_tool_names = [t.name for t in db_tools]
            print(f"\n[DB tools from MCP] {db_tool_names}\n")

            llm = ChatAnthropic(model=MODEL, max_tokens=4096)
            agent = create_agent(llm, all_tools, system_prompt=_build_system_prompt())

            inputs = {"messages": [HumanMessage(content=user_message)]}
            final_text = ""

            async for step in agent.astream(inputs, stream_mode="updates"):
                for node_name, node_output in step.items():
                    for msg in node_output.get("messages", []):

                        if isinstance(msg, AIMessage):
                            usage = msg.response_metadata.get("usage", {})
                            if usage:
                                print(
                                    f"[{node_name}] "
                                    f"in={usage.get('input_tokens','?')} "
                                    f"out={usage.get('output_tokens','?')}"
                                )
                            for tc in msg.tool_calls:
                                print(f"  → {tc['name']}({tc['args']})")
                            if isinstance(msg.content, str) and msg.content:
                                final_text = msg.content

                        elif isinstance(msg, ToolMessage):
                            preview = msg.content[:200] + "…" if len(msg.content) > 200 else msg.content
                            print(f"    ↳ [{msg.name}] {preview}")

            print(f"\nAssistant: {final_text}")
            return final_text


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Hi! What do you know about me, and what data do you have access to?"
    )
    asyncio.run(run_agent(query))
