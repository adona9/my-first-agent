"""
Persistent Memory Agent — Tutorial

Demonstrates how to give an agent memory that survives across separate runs.

──────────────────────────────────────────────────────────────────────────────
THE CORE IDEA
──────────────────────────────────────────────────────────────────────────────
The API is stateless — it has no built-in memory. Every call starts fresh.
Persistent memory is an application-level concern: you store facts somewhere,
load them at the start of each conversation, and inject them into the context.

There are two common patterns:

  1. Full injection (this example)
     Load ALL memories and put them in the system prompt. Simple, and works
     well for dozens to a few hundred short facts.

  2. Retrieval-augmented (for scale)
     Embed memories as vectors and retrieve only the most relevant ones.
     Used when you have thousands of memories or they're long.

──────────────────────────────────────────────────────────────────────────────
HOW IT WORKS
──────────────────────────────────────────────────────────────────────────────
  1. At startup, all memories are loaded from agent_memories.json and
     injected into the system prompt. Claude "knows" them from the start.

  2. During the conversation Claude has three memory tools:
       save_memory   — persist a new fact to disk
       search_memory — search existing memories by keyword
       delete_memory — remove an incorrect or outdated memory

  3. The agent decides on its own when to save something. You can also
     tell it explicitly: "remember that..." or "forget that...".

  4. On the next run, the saved facts are loaded again. The agent picks up
     exactly where it left off.

──────────────────────────────────────────────────────────────────────────────
TRY IT
──────────────────────────────────────────────────────────────────────────────
  Run 1: python memory_agent.py "My name is Alessandro and I prefer metric units."
  Run 2: python memory_agent.py "What do you know about me?"
  Run 3: python memory_agent.py "Actually I prefer imperial units. Update that."
  Run 4: python memory_agent.py "What unit system do I prefer?"

  Inspect the memory file at any point:
    cat agent_memories.json
"""

import json
import sys

import anthropic

from memory_store import delete_memory, format_for_prompt, save_memory, search_memories

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL = "claude-haiku-4-5"

# ── Tool definitions ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "save_memory",
        "description": (
            "Persist a fact to long-term memory. Call this whenever the user shares "
            "something worth remembering across conversations: preferences, names, "
            "personal details, past decisions, stated goals, etc. "
            "Be concise — one clear fact per memory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact to remember, written as a concise statement.",
                }
            },
            "required": ["content"],
        },
    },
    {
        "name": "search_memory",
        "description": (
            "Search existing memories by keyword. Use this when the user asks about "
            "something that might have been mentioned before, or before saving a new "
            "memory to check if a similar one already exists."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or phrase to search for.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "delete_memory",
        "description": (
            "Delete a memory by its id. Use this when the user corrects something "
            "or asks you to forget a specific fact. Each memory has a short id shown "
            "in brackets, e.g. [a1b2c3]. Search first to find the right id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The 6-character memory id to delete.",
                }
            },
            "required": ["memory_id"],
        },
    },
]

TOOL_DISPATCH = {
    "save_memory":   lambda args: save_memory(**args),
    "search_memory": lambda args: search_memories(**args),
    "delete_memory": lambda args: delete_memory(**args),
}


def execute_tool(name: str, tool_input: dict) -> str:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(tool_input)
    except TypeError as e:
        return f"Error: bad arguments for '{name}': {e}"


# ── System prompt (assembled fresh each run) ───────────────────────────────────
# Key design decision: the system prompt is built at runtime, not hardcoded.
#
# format_for_prompt() reads agent_memories.json and returns a formatted block
# like this (empty string if no memories exist yet):
#
#   ## What you remember from previous conversations
#   - [a1b2c3] User's name is Alessandro
#   - [d4e5f6] User prefers metric units
#
# Injecting memories here means Claude has them from turn 1 of every
# conversation — no tool call needed to "load" them.

def build_system_prompt() -> str:
    memory_block = format_for_prompt()

    base = """\
You are a helpful personal assistant with long-term memory.

You can remember facts across conversations using your memory tools:
  • save_memory   — save something new worth keeping
  • search_memory — search what you already know
  • delete_memory — remove or correct an outdated memory

When the user shares a personal detail, preference, or important fact,
save it proactively. When they ask to update something, delete the old
memory and save the corrected version."""

    if memory_block:
        return base + "\n\n" + memory_block
    return base


# ── Agentic loop ───────────────────────────────────────────────────────────────

def run_agent(user_message: str) -> str:
    client = anthropic.Anthropic()
    system = build_system_prompt()
    messages = [{"role": "user", "content": user_message}]

    print(f"\nUser: {user_message}\n")

    # Show which memories were loaded — useful for understanding what the
    # agent already knows before the conversation even starts.
    memory_block = format_for_prompt()
    if memory_block:
        print("[Loaded into context]\n" + memory_block + "\n")
    else:
        print("[No memories yet]\n")

    iteration = 0
    while True:
        iteration += 1
        print(f"[API call #{iteration}]")

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        print(f"  stop_reason : {response.stop_reason}")
        print(f"  input_tokens: {response.usage.input_tokens}, output_tokens: {response.usage.output_tokens}")

        if response.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            print(f"\nAssistant: {final_text}")
            return final_text

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  → {block.name}({json.dumps(block.input)})")
                    result = execute_tool(block.name, block.input)
                    print(f"    ↳ {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
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
        else "Hi! What do you know about me so far?"
    )
    run_agent(query)
