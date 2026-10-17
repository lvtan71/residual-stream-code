# How Coding Agents Work: the full code

The scripts behind [How Coding Agents Work: Claude Code and Cursor, from Their Documentation](https://residual-stream.lvtan71.workers.dev/posts/how-coding-agents-work/). Each `.out` file is what its script printed for the post.

| Script | What it does | Needs |
| --- | --- | --- |
| `guard.py` | A `PreToolUse` hook for Claude Code: reads the pending tool call as JSON on stdin and exits with code 2 to block a recursive force delete or a force push. It prints nothing when allowed, so it has no `.out`. | Python, instant |
| `try_guard.py` | Feeds `guard.py` eight pending `Bash` calls, as Claude Code would, and prints which it allows and which it blocks. | Python, instant |

To use the hook in a project, copy `guard.py` to `.claude/hooks/guard.py` and register it in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": "python3 .claude/hooks/guard.py" }] }
    ]
  }
}
```

The whole folder runs in a second on a CPU. The outputs here came from an Apple M4 laptop; the scripts need only the Python standard library.

```bash
python try_guard.py
```
