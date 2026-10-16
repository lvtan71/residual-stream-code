"""Run the notebooks and save each one with its outputs.

Each notebook runs in notebooks/, the way it runs after a clone, and is
written back in place with its outputs, so a reader who opens it on GitHub
or in Colab sees the latest run before running anything. A note under the
title says where and when that run happened. Each code cell's output is
compared with the "The post printed" cell after it, timings masked, and the
comparison goes to runs/<slug>.json.

usage: python tools/run_notebooks.py [notebooks/<slug>.ipynb ...] [--timeout SECONDS]
(default: every notebook; the default timeout is per cell, 6 hours)
"""

import argparse
import datetime
import glob
import json
import os
import platform
import re
import subprocess
import time

import nbformat
from nbclient import NotebookClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTE = "**Outputs saved from a run on"
MASK = re.compile(r"\b\d[\d,.]*\s?(?:s|ms|us|µs|min|h)\b")


def machine():
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        gpu = None
    if not gpu and platform.system() == "Darwin":
        gpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return gpu or platform.processor() or platform.machine()


def compare(nb):
    rows = []
    for i, c in enumerate(nb.cells):
        if c.cell_type != "markdown" or not c.source.startswith("The post printed:"):
            continue
        exp = c.source.split("```text\n", 1)[1].rsplit("\n```", 1)[0]
        out = "".join(o.get("text", "") for o in nb.cells[i - 1].get("outputs", [])
                      if o.get("output_type") == "stream" and o.get("name") == "stdout")
        want = [MASK.sub("<t>", l.rstrip()) for l in exp.split("\n") if l.strip()]
        got = {MASK.sub("<t>", l.rstrip()) for l in out.split("\n")}
        rows.append({"cell": i - 1, "lines": len(want), "matched": sum(l in got for l in want),
                     "missing": [l for l in want if l not in got][:20],
                     "printed": out[-4000:]})
    return rows


def run(path, timeout):
    slug = os.path.basename(path)[:-6]
    nb = nbformat.read(path, as_version=4)
    nb.cells = [c for c in nb.cells if not (c.cell_type == "markdown" and c.source.startswith(NOTE))]
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=True,
                            resources={"metadata": {"path": os.path.dirname(os.path.abspath(path))}})
    t = time.time()
    crash = None
    try:
        client.execute()
    except Exception as e:  # a kernel death or a cell timeout; the outputs so far are kept
        crash = str(e).strip().split("\n")[-1][:300]
    minutes = (time.time() - t) / 60
    errors = [f"cell {i}: {o['ename']}: {o['evalue'][:200]}" for i, c in enumerate(nb.cells)
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    # Local paths of the machine that ran it are noise to a reader.
    for c in nb.cells:
        for o in c.get("outputs", []):
            if "text" in o:
                o["text"] = o["text"].replace(ROOT, "residual-stream-code")
    try:
        import torch
        versions = f"torch {torch.__version__}"
    except ImportError:
        versions = ""
    when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    status = "it stopped with an error" if (errors or crash) else "every cell ran"
    nb.cells.insert(1, nbformat.v4.new_markdown_cell(
        f"{NOTE} {machine()} on {when}** ({versions + ', ' if versions else ''}"
        f"{minutes:.0f} min; {status}). Run the notebook to replace them with your own."))
    nbformat.write(nb, path)
    rows = compare(nb)
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    report = {"slug": slug, "machine": machine(), "date": when, "minutes": round(minutes, 1),
              "crash": crash, "errors": errors, "cells": rows}
    json.dump(report, open(os.path.join(ROOT, "runs", slug + ".json"), "w"), indent=1)
    matched = sum(r["matched"] for r in rows)
    total = sum(r["lines"] for r in rows)
    print(f"{slug}: {minutes:.1f} min, {matched}/{total} printed lines reproduced"
          + (f", CRASH {crash}" if crash else "") + (f", {len(errors)} error(s): {errors[0]}" if errors else ""),
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    a = ap.parse_args()
    for p in a.paths or sorted(glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb"))):
        run(p, a.timeout)
    print("finished", flush=True)
