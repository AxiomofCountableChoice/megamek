import subprocess
import os
import time
import signal
import sys

# Change this to whatever port you'd like the MegaMek RLServer to listen on
PORT = 2348

# The root megamek repository directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "megamek"))

# Mech files are located in mm-data
mm_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mm-data"))

print(f"Starting MegaMek Offline Trainer from CWD: {repo_root}")

cmd = [
    "build/install/MegaMek/bin/megamek",
    "-rlexport",
    "-autogen",
    "-randomMap",
    "-p1meks", os.path.join(mm_data_root, "data", "mekfiles", "meks", "3075", "Pariah Prime.mtf"),
    "-p2meks", os.path.join(mm_data_root, "data", "mekfiles", "meks", "Rec Guides ilClan", "Vol 33", "Gyrfalcon 5.mtf")
]

# Set environment variable to ensure we connect properly
env = os.environ.copy()
env["RL_SERVER_PORT"] = str(PORT)

# Redirect stdout and stderr so we can stream it and see Java crashes
proc = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

import threading
def stream_output(pipe):
    try:
        for line in iter(pipe.readline, ''):
            print(f"[MegaMek] {line}", end='')
    except Exception:
        pass
threading.Thread(target=stream_output, args=(proc.stdout,), daemon=True).start()

def signal_handler(sig, frame):
    print("\nShutting down MegaMek Offline Trainer...")
    proc.send_signal(signal.SIGTERM)
    # Also attempt to kill the Java process directly since Gradle script sometimes orphans it
    subprocess.run(["pkill", "-f", "java.*MegaMek"], stderr=subprocess.DEVNULL)
    proc.wait(timeout=2)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

print(f"MegaMek Started (PID {proc.pid}). Press Ctrl+C to terminate.")

# Wait for it to finish
try:
    proc.wait()
except KeyboardInterrupt:
    signal_handler(signal.SIGINT, None)
