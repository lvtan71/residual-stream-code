import json
import subprocess
import sys

# Feed the hook pending tool calls, as Claude Code would.
for command in ("npm test", "rm -rf build/", "rm -fr build/", "rm -r -f build/", "rm build/old.log",
                "git push --force-with-lease", "git push -f origin main", "python -c 'import shutil; shutil.rmtree(\"build\")'"):
    event = {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": command}}
    r = subprocess.run([sys.executable, "guard.py"], input=json.dumps(event), capture_output=True, text=True)
    print(f"{command!r:58s} -> {'allowed' if r.returncode == 0 else 'blocked (exit 2)'}")
