import re

# distill_trace.out (seed 0, with the held-out KL to the teacher logged) as a table.
print("seed 0, held-out blog: bits per byte / KL to the teacher over its top 64 (nats)")
print(f"{'labels':10s}" + "".join(f"{'step ' + str(s):>15s}" for s in (0, 50, 100, 200, 400)))
for line in open("distill_trace.out"):
    m = re.match(r"^(hard|soft|mix)", line)
    if m:
        pts = dict((int(s), (b, k)) for s, b, k in
                   re.findall(r"step (\d+): blog ([\d.]+), wiki [\d.]+, KL to teacher ([\d.]+)", line))
        label = {"mix": "half each"}.get(m.group(1), m.group(1))
        print(f"{label:10s}" + "".join(f"{pts[s][0] + ' / ' + pts[s][1]:>15s}" for s in (0, 50, 100, 200, 400)))
