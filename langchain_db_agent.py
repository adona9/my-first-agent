"""
LangChain DB Agent — Tutorial

Solves the same problem as db_agent.py using LangChain + LangGraph.
Reading both side-by-side is the best way to understand the trade-offs.

──────────────────────────────────────────────────────────────────────────────
WHEN TO USE THE RAW SDK (what you've been doing)
──────────────────────────────────────────────────────────────────────────────
  • You want full visibility and control over every API call
  • You're debugging or learning how the protocol works
  • You need to optimise costs / latency precisely
  • Your tool loop has unusual branching logic

WHEN TO USE LANGCHAIN / LANGGRAPH
──────────────────────────────────────────────────────────────────────────────
  • You want to swap models (Claude → GPT-4 → Gemini) without rewriting tools
  • You need a prompt pipeline with variable substitution (ChatPromptTemplate)
  • You want conversation memory managed for you (in-memory or persistent)
  • You want a pre-built ReAct agent loop instead of writing one yourself
  • You're building complex multi-agent graphs (LangGraph shines here)

──────────────────────────────────────────────────────────────────────────────
KEY DIFFERENCES YOU'LL SEE BELOW vs db_agent.py
──────────────────────────────────────────────────────────────────────────────
  1. @tool decorator  — no manual JSON schema; docstring becomes the description
  2. ChatAnthropic    — LangChain's model wrapper; same model, cleaner API
  3. llm.bind_tools() — attaches tools to the model in one line
  4. create_agent     — a pre-built ReAct loop; you don't write the while-loop
  5. graph.stream()   — iterate over every step of the agent as it runs

Run it:
  python langchain_db_agent.py "Which director has the most movies in the database?"
  python langchain_db_agent.py "What are the top 5 stocks by market cap?"
"""

import sys

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

from db_tools import describe_table, list_tables, run_query

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"

# ── Tool definitions ───────────────────────────────────────────────────────────
# Compare this to the TOOLS list in db_agent.py.
#
# Raw SDK: you write a JSON schema dict by hand — verbose but explicit.
# LangChain: the @tool decorator + type hints + docstring do the same job.
#   - The function name  → tool name
#   - The docstring      → tool description (Claude reads this)
#   - The type hints     → input schema (LangChain generates JSON schema for you)
#
# Both approaches produce the same thing under the hood; LangChain just
# saves you from writing repetitive boilerplate.

@tool
def list_tables_tool() -> str:
    """List all tables in the research database along with their row counts.
    Always call this first when you don't know what data is available."""
    return list_tables()


@tool
def describe_table_tool(table_name: str) -> str:
    """Show the column names, types, and a few sample rows for a table.
    Call this before writing a query so you know the exact column names.

    Args:
        table_name: Name of the table to describe.
    """
    return describe_table(table_name)


@tool
def run_query_tool(sql: str) -> str:
    """Execute a SQL SELECT query against the research database and get results
    as a JSON array. If the query fails the result starts with 'Error:' — fix
    the SQL and retry. Only SELECT queries are allowed.

    Args:
        sql: A valid SQLite SELECT statement.
    """
    return run_query(sql)


TOOLS = [list_tables_tool, describe_table_tool, run_query_tool]

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM = """\
You are a data research assistant with access to a SQLite database.
To answer questions, you should:
  1. Call list_tables_tool to see what's available.
  2. Call describe_table_tool on the relevant table(s) to learn column names.
  3. Write and run a SELECT query with run_query_tool.
  4. If run_query_tool returns a result starting with "Error:", fix the SQL and retry.
  5. Summarize the query results in plain English as your final answer.

Always base your answers on actual query results — do not guess."""


# ── Agent setup ────────────────────────────────────────────────────────────────
# ChatAnthropic is LangChain's wrapper around the Anthropic API.
# It has the same parameters you know — model, max_tokens, temperature, etc.
#
# create_agent builds the ReAct loop for you:
#   "Reason → Act (call a tool) → Observe (get result) → repeat"
# This is exactly the while-loop you wrote by hand in db_agent.py, but
# packaged as a LangGraph graph you can inspect, extend, or visualise.

def build_agent():
    llm = ChatAnthropic(model=MODEL, max_tokens=4096)

    # create_react_agent returns a CompiledGraph.
    # The system prompt goes in as a plain string here; LangGraph wraps it.
    return create_agent(llm, TOOLS, system_prompt=SYSTEM)


# ── Run the agent ──────────────────────────────────────────────────────────────
# graph.stream() yields one dict per step in the graph:
#   {"agent":  {"messages": [AIMessage(...)]}}       ← model decided to act
#   {"tools":  {"messages": [ToolMessage(...)]}}     ← tool result came back
#
# Each AIMessage may contain text (thinking) and/or tool_calls.
# Each ToolMessage contains the tool's return value.
#
# Compare to db_agent.py where you inspect response.stop_reason and
# iterate over response.content manually — same information, different shape.

def run_agent(user_message: str) -> str:
    agent = build_agent()

    print(f"\nQuestion: {user_message}\n")

    inputs = {"messages": [HumanMessage(content=user_message)]}
    final_text = ""

    for step in agent.stream(inputs, stream_mode="updates"):
        for node_name, node_output in step.items():
            messages = node_output.get("messages", [])

            for msg in messages:
                # ── Agent node: model produced a response ──────────────────
                if isinstance(msg, AIMessage):
                    # token usage is attached to the response_metadata dict
                    usage = msg.response_metadata.get("usage", {})
                    if usage:
                        print(
                            f"[{node_name}] "
                            f"input_tokens={usage.get('input_tokens', '?')}, "
                            f"output_tokens={usage.get('output_tokens', '?')}"
                        )

                    # tool_calls is a list of dicts: {name, args, id}
                    for tc in msg.tool_calls:
                        print(f"  → {tc['name']}({tc['args']})")

                    # plain text means Claude is giving its final answer
                    if msg.content and isinstance(msg.content, str) and msg.content:
                        final_text = msg.content

                # ── Tools node: a tool returned a result ───────────────────
                elif isinstance(msg, ToolMessage):
                    preview = msg.content[:300] + "…" if len(msg.content) > 300 else msg.content
                    print(f"    ↳ [{msg.name}] {preview}")

    print(f"\nAnswer: {final_text}")
    return final_text


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What are the top 3 highest-rated movies, and who directed them?"
    )
    run_agent(query)
