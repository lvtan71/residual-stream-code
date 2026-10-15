#!/usr/bin/env python3
# A PreToolUse hook for Claude Code: it receives the pending tool call as
# JSON on stdin; exit code 2 blocks the call, and what it prints to stderr
# is shown to the model as the reason. Exit 0 lets the call proceed.
# A check like this shows the mechanism; it is not a security boundary (a
# command can always be written another way).
import json
import re
import shlex
import sys

def dangerous(command):
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    for i, w in enumerate(words):
        args = words[i + 1:]
        flags = "".join(a.lstrip("-") for a in args if a.startswith("-") and not a.startswith("--"))
        long = {a for a in args if a.startswith("--")}
        if w == "rm" and ("r" in flags.lower() or "--recursive" in long) and ("f" in flags or "--force" in long):
            return True                                          # rm -rf, -fr, -r -f, -Rf, --recursive --force
        if w == "push" and i > 0 and words[i - 1] == "git":
            if "f" in flags or "--force" in long or any(re.match(r"\+", a) for a in args):
                return True                                      # force push (--force-with-lease is allowed)
    return False

event = json.load(sys.stdin)
if event.get("tool_name") == "Bash" and dangerous(event.get("tool_input", {}).get("command", "")):
    print("Blocked by project policy: recursive force delete or force push. Ask the user first.", file=sys.stderr)
    sys.exit(2)
sys.exit(0)
