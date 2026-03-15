"""
File System Agent — Tutorial

Demonstrates the core pattern for building a tool-using agent with the Claude API:
  1. Define tools as JSON schemas (so Claude knows what's available and how to call them)
  2. Run an "agentic loop": call the API, execute any requested tools, feed results back
  3. Repeat until Claude gives a final answer (stop_reason == "end_turn")

Run it:
  python agent.py "list the files in the current directory"
  python agent.py "find all mentions of 'knowledge' in the notes"
  python agent.py "create a file called hello.txt with the content 'Hello, world!'"
"""

import json
import sys

import anthropic

from filesystem_tools import list_directory, read_file, search_files, write_file

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL = "claude-haiku-4-5"

# ── Tool definitions ───────────────────────────────────────────────────────────
# These JSON schemas are sent to Claude with every request.
# Claude reads the "description" fields to decide when and how to use each tool.
# The "input_schema" tells Claude exactly what arguments to provide.
TOOLS = [
    {
        "name": "list_directory",
        "description": (
            "List files and directories at a given path. "
            "Use this to explore the file system and understand what's available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list. Defaults to current directory.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read and return the full contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file. Creates the file (and any parent directories) "
            "if they don't exist, or overwrites it if it does."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search for lines containing a text pattern across files in a directory "
            "(case-insensitive, recursive). Returns matching lines with file path and line number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory to search in.",
                },
                "pattern": {
                    "type": "string",
                    "description": "Text to search for (case-insensitive).",
                },
                "file_glob": {
                    "type": "string",
                    "description": (
                        "Glob pattern to filter which files to search. "
                        "Examples: '*.py', '*.txt'. Defaults to '*' (all files)."
                    ),
                },
            },
            "required": ["directory", "pattern"],
        },
    },
]

# ── Tool dispatch ──────────────────────────────────────────────────────────────
# Maps the tool name Claude uses back to the actual Python function.
TOOL_DISPATCH = {
    "list_directory": list_directory,
    "read_file": read_file,
    "write_file": write_file,
    "search_files": search_files,
}


def execute_tool(name: str, tool_input: dict) -> str:
    """Look up and call a tool by name. Returns a string result."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(**tool_input)
    except TypeError as e:
        return f"Error: bad arguments for tool '{name}': {e}"


# ── Agentic loop ───────────────────────────────────────────────────────────────

def run_agent(user_message: str) -> str:
    """
    Run the agent and return Claude's final text response.

    Key concepts illustrated here:
    - `messages` is the full conversation history, sent on every API call
      (the API is stateless — it knows nothing between calls)
    - `stop_reason == "tool_use"` means Claude wants to call tools
    - `stop_reason == "end_turn"` means Claude is done and giving a final answer
    - Tool results must be sent back as a "user" message with type "tool_result"
    - Each tool_result must reference the matching tool_use block via `tool_use_id`
    """
    client = anthropic.Anthropic()

    # Start with just the user's message
    messages = [{"role": "user", "content": user_message}]

    print(f"\nUser: {user_message}\n")

    iteration = 0
    while True:
        iteration += 1
        print(f"[API call #{iteration}]")

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )

        print(f"  stop_reason: {response.stop_reason}")

        # ── Case 1: Claude is done ─────────────────────────────────────────────
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text = block.text
            print(f"\nAssistant: {final_text}")
            return final_text

        # ── Case 2: Claude wants to use tools ─────────────────────────────────
        if response.stop_reason == "tool_use":
            # IMPORTANT: append Claude's full response to history FIRST.
            # The next user message (tool results) must follow an assistant message
            # that contains the matching tool_use blocks — the API enforces this.
            messages.append({"role": "assistant", "content": response.content})

            # Execute every tool Claude requested and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    args_preview = json.dumps(block.input)
                    print(f"  → {block.name}({args_preview})")

                    result = execute_tool(block.name, block.input)

                    preview = result[:300] + "…" if len(result) > 300 else result
                    print(f"    ↳ {preview}")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,  # links back to the tool_use block
                        "content": result,
                    })

            # Send all tool results back in a single user message
            messages.append({"role": "user", "content": tool_results})

        else:
            # Shouldn't normally happen (e.g. max_tokens, stop_sequence)
            print(f"  Unexpected stop_reason: {response.stop_reason!r}. Stopping.")
            break

    return ""


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "List the files in the current directory, then read any .txt files you find."
    )
    run_agent(query)
