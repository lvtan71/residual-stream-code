import os
import time

# Imported before anything is timed: waits, up to ten minutes, until the
# machine's one-minute load average drops below 2, so that no other heavy job
# shares the CPU or GPU with the timings. It prints nothing. (A reconstruction:
# the original file was lost; it changes no result, only when timing starts.)
deadline = time.time() + 600
while os.getloadavg()[0] >= 2.0 and time.time() < deadline:
    time.sleep(10)
