"""
File-based memory store.

Memories are plain JSON objects persisted in a single file (MEMORY_FILE).
Each memory has a short auto-generated id so the agent can delete or refer
to specific entries.

The file format is a JSON array so it's human-readable and easy to inspect:

  [
    {"id": "a1b2", "timestamp": "2026-03-14T10:00:00", "content": "User prefers metric units"},
    {"id": "c3d4", "timestamp": "2026-03-14T10:05:00", "content": "User's name is Alessandro"}
  ]
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

MEMORY_FILE = Path(".agent_memories.json")


# ── Internal helpers ────────────────────────────────────────────────────────────

def _load() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save(memories: list[dict]) -> None:
    MEMORY_FILE.write_text(json.dumps(memories, indent=2))


def _short_id() -> str:
    return uuid.uuid4().hex[:6]


# ── Public API (called by the agent tools) ──────────────────────────────────────

def save_memory(content: str) -> str:
    """Append a new memory and return a confirmation string."""
    memories = _load()
    entry = {
        "id": _short_id(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content": content,
    }
    memories.append(entry)
    _save(memories)
    return f"Saved memory [{entry['id']}]: {content}"


def search_memories(query: str) -> str:
    """Return memories whose content contains `query` (case-insensitive)."""
    memories = _load()
    q = query.lower()
    matches = [m for m in memories if q in m["content"].lower()]
    if not matches:
        return f"No memories found matching '{query}'."
    lines = [f"[{m['id']}] {m['content']}" for m in matches]
    return "\n".join(lines)


def delete_memory(memory_id: str) -> str:
    """Remove a memory by its id. Returns an error string if not found."""
    memories = _load()
    original_count = len(memories)
    memories = [m for m in memories if m["id"] != memory_id]
    if len(memories) == original_count:
        return f"Error: no memory with id '{memory_id}'"
    _save(memories)
    return f"Deleted memory [{memory_id}]."


def format_for_prompt() -> str:
    """
    Format all memories as a block suitable for injection into the system prompt.
    Returns an empty string if there are no memories yet.
    """
    memories = _load()
    if not memories:
        return ""
    lines = [f"- [{m['id']}] {m['content']}" for m in memories]
    return "## What you remember from previous conversations\n" + "\n".join(lines)
