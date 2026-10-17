import subprocess
import sys

# race.py run both ways: on the bag of words, then with the raw question
# length added as one more feature.
for args in ([], ["length"]):
    subprocess.run([sys.executable, "race.py", *args], check=True)
