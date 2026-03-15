"""
File system tools for the agent.

These are plain Python functions — no Claude API involved here.
The agent wraps these in JSON schema definitions that Claude understands,
then calls them when Claude decides to use a tool.
"""

import fnmatch
from pathlib import Path


def list_directory(path: str = ".") -> str:
    """List files and directories at the given path."""
    p = Path(path).expanduser().resolve()

    if not p.exists():
        return f"Error: path '{path}' does not exist"
    if not p.is_dir():
        return f"Error: '{path}' is not a directory"

    entries = []
    for item in sorted(p.iterdir()):
        if item.is_dir():
            entries.append(f"[dir]  {item.name}/")
        else:
            size = item.stat().st_size
            entries.append(f"[file] {item.name} ({size} bytes)")

    if not entries:
        return f"Directory '{p}' is empty."
    return f"Contents of {p}:\n" + "\n".join(entries)


def read_file(path: str) -> str:
    """Read the contents of a file."""
    p = Path(path).expanduser().resolve()

    if not p.exists():
        return f"Error: file '{path}' does not exist"
    if not p.is_file():
        return f"Error: '{path}' is not a file"
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: '{path}' appears to be a binary file (not readable as text)"
    except PermissionError:
        return f"Error: permission denied reading '{path}'"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed."""
    p = Path(path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{p}'"
    except PermissionError:
        return f"Error: permission denied writing to '{path}'"
    except OSError as e:
        return f"Error: {e}"


def search_files(directory: str, pattern: str, file_glob: str = "*") -> str:
    """
    Search for lines containing `pattern` in files under `directory`.
    - pattern: text to search for (case-insensitive)
    - file_glob: which files to search, e.g. '*.py', '*.txt' (default: all files)
    """
    d = Path(directory).expanduser().resolve()

    if not d.exists():
        return f"Error: directory '{directory}' does not exist"
    if not d.is_dir():
        return f"Error: '{directory}' is not a directory"

    results = []
    for file_path in sorted(d.rglob(file_glob)):
        if not file_path.is_file():
            continue
        try:
            for i, line in enumerate(
                file_path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
            ):
                if pattern.lower() in line.lower():
                    results.append(f"{file_path}:{i}: {line.rstrip()}")
        except (PermissionError, OSError):
            pass

    if not results:
        return f"No matches for '{pattern}' in '{d}' (glob: {file_glob})"
    return f"Found {len(results)} match(es):\n" + "\n".join(results)
