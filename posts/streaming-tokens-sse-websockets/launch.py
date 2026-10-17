import socket
import subprocess
import sys
import time

# The post ran the server and the proxy in terminals of their own. The
# scripts here start them in the background instead, with their log lines
# hidden, and wait until they listen (the server loads the model first).
def start(*args, port):
    p = subprocess.Popen([sys.executable, *args], stdout=subprocess.DEVNULL)
    while True:
        if p.poll() is not None:
            raise RuntimeError(f"{args[0]} exited")
        try:
            socket.create_connection(("127.0.0.1", port)).close()
            return p
        except OSError:
            time.sleep(1)
