# my-first-agent

A series of progressively more advanced agent examples, each building on the concepts introduced before it. Written against the Anthropic SDK directly before introducing frameworks, so the underlying mechanics are clear.

## Setup

```bash
# Install dependencies
uv sync

# Seed the SQLite database (required for all DB examples)
uv run python seed_db.py
```

Set your API key:
```bash
export ANTHROPIC_API_KEY=sk-...
```

---

## Examples

### 1. `filesystem_agent.py` — The core loop

**Start here.** Introduces the fundamental pattern every agent is built on.

```bash
uv run python filesystem_agent.py "list the files in the current directory"
uv run python filesystem_agent.py "find all mentions of 'knowledge' in the notes"
```

**What it teaches:**
- Tools are JSON schemas sent to Claude on every request; the `description` field is what Claude reads to decide when to call them
- The API is stateless — you maintain the conversation history yourself and send it in full on every call
- `stop_reason == "tool_use"` means Claude wants to call a tool; `stop_reason == "end_turn"` means it's done
- Tool results must be sent back as a `"user"` message with `type: "tool_result"`, referencing the matching `tool_use_id`

**Files:** `filesystem_agent.py`, `filesystem_tools.py`

---

### 2. `db_agent.py` — Structured output and error recovery

Same loop, harder problem. Claude has to reason about what data exists before it can query it.

```bash
uv run python db_agent.py "Which director has the most movies in the database?"
uv run python db_agent.py "What are the top 5 stocks by market cap?"
uv run python db_agent.py "Which continent has the highest average GDP per country?"
```

**What it teaches:**
- Tools can return structured JSON — Claude parses and reasons over it
- When a tool returns an error string, Claude sees it and retries with corrected input (no special handling needed — it's just text)
- `client.messages.count_tokens()` lets you estimate input tokens before making a call
- `response.usage` shows actual input/output token counts after the call

**Files:** `db_agent.py`, `db_tools.py`, `seed_db.py`

---

### 3. `langchain_db_agent.py` — The same agent, with LangChain

Solves the same DB problem as example 2, rewritten with LangChain + LangGraph. Reading both side-by-side is the best way to understand the trade-offs.

```bash
uv run python langchain_db_agent.py "Which director has the most movies?"
```

**What it teaches:**

| Concept | Raw SDK | LangChain |
|---|---|---|
| Tool definitions | Hand-written JSON schemas | `@tool` decorator — docstring + type hints generate the schema |
| Model | `anthropic.Anthropic().messages.create(...)` | `ChatAnthropic(model=...)` |
| Agent loop | Manual `while True` | `create_agent(llm, tools)` — loop is built for you |
| Observability | `response.stop_reason`, `response.usage` | `graph.stream()` — one dict per step |

**When to use LangChain over the raw SDK:**
- You need to swap models (Claude → GPT-4) without rewriting tool logic
- You want prompt templates with variable substitution
- You want conversation memory managed automatically
- You're building multi-agent graphs (LangGraph shines here)

**Files:** `langchain_db_agent.py`

---

### 4. `memory_agent.py` — Persistent memory across runs

An agent that remembers things between separate invocations by saving facts to a JSON file.

```bash
# Run these in sequence to see memory persist and update
uv run python memory_agent.py "My name is Alessandro and I prefer metric units."
uv run python memory_agent.py "What do you know about me?"
uv run python memory_agent.py "Actually I prefer imperial units. Update that."
uv run python memory_agent.py "What unit system do I prefer?"

# Inspect the memory file directly
cat .agent_memories.json
```

**What it teaches:**

The API has no built-in memory — persistence is entirely your responsibility. The pattern here is:

1. **Load** all memories from disk at startup
2. **Inject** them into the system prompt so Claude knows them from turn 1
3. Give Claude tools to **save**, **search**, and **delete** memories during the conversation
4. On the next run, repeat from step 1

This is called **full injection** — suitable for dozens to a few hundred short facts. The alternative is **retrieval-augmented memory**: embed memories as vectors and retrieve only the most relevant ones, used when you have thousands of memories or they're long.

Claude manages its own memory: it decides what's worth saving, searches before overwriting, and deletes outdated facts when correcting them.

**Files:** `memory_agent.py`, `memory_store.py`

---

### 5. `mcp_agent.py` — Tools via MCP server

The same DB agent again, but the tools now live in a separate server process. The agent discovers them at runtime over the MCP protocol instead of having them hardcoded.

```bash
uv run python mcp_agent.py "Which director has the most movies in the database?"
```

**What it teaches:**

MCP (Model Context Protocol) decouples tool *implementation* from the agent that uses it.

| | `db_agent.py` | `mcp_agent.py` |
|---|---|---|
| Tool definitions | Hardcoded JSON schemas in the agent file | None — discovered at runtime via `session.list_tools()` |
| Tool execution | Direct Python function call | `session.call_tool(name, args)` over JSON-RPC |
| Coupling | Tools and agent in the same process | Server and agent are separate processes |

The transport is **stdio**: the client starts `mcp_db_server.py` as a subprocess and talks to it over stdin/stdout. You can swap this for HTTP/SSE to connect to a remote server with no changes to the agent loop.

The same MCP server can be consumed by Claude Code, Cursor, a LangChain agent, or any other MCP-compatible client — without changing a line of server code.

**Files:** `mcp_agent.py`, `mcp_db_server.py`

---

### 6. `combined_agent.py` — Everything together

Combines LangChain, persistent memory, and MCP into a single agent. A general-purpose assistant that can query the database and remember personal context across runs.

```bash
uv run python combined_agent.py "My name is Alessandro. What tables do you have access to?"
uv run python combined_agent.py "Who directed the most movies? Also, what do you know about me?"
uv run python combined_agent.py "What are the top 3 stocks by market cap?"
```

**What it teaches:**

The main integration challenge is that each system uses a different tool abstraction:

| System | Tool format |
|---|---|
| Raw SDK | Plain dicts: `{"name": ..., "input_schema": ...}` |
| LangChain | `BaseTool` instances: `@tool` or `StructuredTool` |
| MCP | `mcp.types.Tool`: `.name`, `.description`, `.inputSchema` |

The solution is `_mcp_tool_to_langchain()`: convert each MCP tool into a LangChain `StructuredTool` by (1) building a Pydantic model from its JSON Schema, and (2) wrapping `session.call_tool()` as an async function. The resulting tools are indistinguishable from `@tool`-decorated functions as far as `create_agent` is concerned.

A second consequence of mixing async MCP tools with LangChain: `agent.stream()` (sync) can't invoke async tools, so you must use `agent.astream()` instead — which is fine since the whole agent runs inside `async def run_agent()`.

**Files:** `combined_agent.py` (+ `mcp_db_server.py`, `memory_store.py`)

---

## Concept map

```
filesystem_agent.py     ← start here: the core agent loop
       │
       ▼
db_agent.py             ← structured output, error recovery, token counting
       │
       ├──► langchain_db_agent.py   ← same problem, framework abstractions
       │
       ├──► memory_agent.py         ← persistence across runs
       │
       ├──► mcp_agent.py            ← tools in a separate server process
       │
       └──► combined_agent.py       ← LangChain + memory + MCP together
```

## Project structure

```
my-first-agent/
├── filesystem_agent.py     # Example 1: core loop
├── filesystem_tools.py
├── db_agent.py             # Example 2: structured output + error recovery
├── db_tools.py
├── seed_db.py              # Creates research.db (movies, countries, stocks)
├── langchain_db_agent.py   # Example 3: LangChain + LangGraph
├── memory_agent.py         # Example 4: persistent memory
├── memory_store.py
├── mcp_db_server.py        # Example 5+6 (server): tools via MCP
├── mcp_agent.py            # Example 5 (client): raw SDK + MCP
├── combined_agent.py       # Example 6: LangChain + memory + MCP
└── pyproject.toml
```
