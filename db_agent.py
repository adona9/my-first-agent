"""
SQLite Research Agent — Tutorial

Demonstrates three additional concepts on top of filesystem_agent.py:
  1. Structured tool output  — tools return JSON; Claude parses and reasons over it
  2. Error recovery          — bad SQL produces "Error: ..." → Claude rewrites and retries
  3. Result summarization    — Claude turns raw query results into a readable answer

Run it (seed the DB first if you haven't):
  python seed_db.py
  python db_agent.py "Which director has the most movies in the database?"
  python db_agent.py "What are the top 5 stocks by market cap?"
  python db_agent.py "Which continent has the highest average GDP per country?"
"""

import json
import sys

import anthropic

from db_tools import describe_table, list_tables, run_query

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"

# ── Tool definitions ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "list_tables",
        "description": (
            "List all tables in the research database along with their row counts. "
            "Always call this first when you don't know what data is available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "describe_table",
        "description": (
            "Show the column names, types, and a few sample rows for a table. "
            "Call this before writing a query so you know the exact column names."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "table_name": {
                    "type": "string",
                    "description": "Name of the table to describe.",
                }
            },
            "required": ["table_name"],
        },
    },
    {
        "name": "run_query",
        "description": (
            "Execute a SQL SELECT query against the research database and get results "
            "as a JSON array. "
            "If the query fails the result starts with 'Error:' — fix the SQL and retry. "
            "Only SELECT queries are allowed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid SQLite SELECT statement.",
                }
            },
            "required": ["sql"],
        },
    },
]

# ── Tool dispatch ──────────────────────────────────────────────────────────────
TOOL_DISPATCH = {
    "list_tables": lambda args: list_tables(),
    "describe_table": lambda args: describe_table(**args),
    "run_query": lambda args: run_query(**args),
}


def execute_tool(name: str, tool_input: dict) -> str:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(tool_input)
    except TypeError as e:
        return f"Error: bad arguments for tool '{name}': {e}"


# ── Agentic loop ───────────────────────────────────────────────────────────────

SYSTEM = """\
You are a data research assistant with access to a SQLite database.
To answer questions, you should:
  1. Call list_tables to see what's available.
  2. Call describe_table on the relevant table(s) to learn column names.
  3. Write and run a SELECT query with run_query.
  4. If run_query returns a result starting with "Error:", fix the SQL and try again.
  5. Summarize the query results in plain English as your final answer.

Always base your answers on actual query results — do not guess."""


def run_agent(user_message: str) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_message}]

    print(f"\nQuestion: {user_message}\n")

    iteration = 0
    while True:
        iteration += 1
        print(f"[API call #{iteration}]")

        token_count = client.messages.count_tokens(
            model=MODEL,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        print(f"  input_tokens (estimated): {token_count.input_tokens}")

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        print(f"  stop_reason: {response.stop_reason}")
        print(f"  input tokens: {response.usage.input_tokens}, output tokens: {response.usage.output_tokens}")

        # ── Case 1: Claude is done ─────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text = block.text
            print(f"\nAnswer: {final_text}")
            return final_text

        # ── Case 2: Claude wants to use a tool ────────────────────────────────
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                print(f"  block type: {block.type}")
                if block.type == "text":
                    print(f"  {block.text}")
                if block.type == "tool_use":
                    args_str = json.dumps(block.input)
                    print(f"  → {block.name}({args_str})")

                    result = execute_tool(block.name, block.input)

                    # Key teaching moment: errors come back as plain strings
                    # starting with "Error:" — Claude will see them and retry.
                    is_error = result.startswith("Error:")
                    preview = result[:300] + "…" if len(result) > 300 else result
                    prefix = "ERROR" if is_error else "result"
                    print(f"    ↳ [{prefix}] {preview}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                        # Flagging errors helps Claude distinguish them from
                        # legitimate empty-result responses.
                        **({"is_error": True} if is_error else {}),
                    })

            messages.append({"role": "user", "content": tool_results})

        else:
            print(f"  Unexpected stop_reason: {response.stop_reason!r}. Stopping.")
            break

    return ""


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "What are the top 3 highest-rated movies, and who directed them?"
    )
    run_agent(query)
