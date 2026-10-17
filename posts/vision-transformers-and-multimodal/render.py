# macOS only: this calls headless Chrome at its macOS install path. The
# folder ships its output as clip-diagram-renders.tar.gz, so other machines
# can skip it. Run it in a folder holding the blog snapshot's images/
# (tar xzf blog-11f17f9.tar.gz).
import glob
import os
import re
import subprocess
import tempfile
# Render each diagram whole, fitted (aspect kept) into a 512x512 white
# square, with headless Chrome; and a second set with all text hidden.
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
os.makedirs("fit", exist_ok=True); os.makedirs("notext", exist_ok=True)
for path in sorted(glob.glob("images/diagrams/*-light.svg")):
    if "-vi-" in path:
        continue
    name = os.path.basename(path)[:-len("-light.svg")]
    svg = open(path).read()
    svg = re.sub(r"<svg([^>]*?)\swidth=\"[^\"]*\"", r"<svg\1", svg, count=1)
    svg = re.sub(r"<svg([^>]*?)\sheight=\"[^\"]*\"", r"<svg\1", svg, count=1)
    svg = svg.replace("<svg", '<svg width="512" height="512" preserveAspectRatio="xMidYMid meet"', 1)
    for folder, extra in (("fit", ""), ("notext", "svg text, svg foreignObject { display: none !important; }")):
        html = (f"<html><body style='margin:0;background:#fff'><style>{extra}</style>"
                f"<div style='width:512px;height:512px'>{svg}</div></body></html>")
        open(f"{tempfile.gettempdir()}/r_{name}.html", "w").write(html)
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=512,512",
                        f"--screenshot={os.getcwd()}/{folder}/{name}.png", f"file://{tempfile.gettempdir()}/r_{name}.html"],
                       capture_output=True, timeout=60)
print(len(glob.glob("fit/*.png")), len(glob.glob("notext/*.png")))
