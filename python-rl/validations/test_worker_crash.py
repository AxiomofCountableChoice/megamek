import subprocess
import time
import os

cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "megamek"))
java_home = os.environ.get("JAVA_HOME", "/home/stuart_hatzioannou/jdk-21.0.2")
env = os.environ.copy()
env["JAVA_HOME"] = java_home
env["PATH"] = f"{os.path.join(java_home, 'bin')}:{env.get('PATH', '')}"
env["RL_SERVER_PORT"] = "4000"

print("Starting MegaMek server on port 4000 (RL Port 5000)...")
server_cmd = [
    os.path.join(java_home, "bin", "java"),
    "-Xmx4096m",
    "-jar", "build/install/MegaMek/MegaMek.jar",
    "-rlexport", "-autogen"
]
server_proc = subprocess.Popen(server_cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Waiting 15 seconds for server to initialize...")
time.sleep(15)

print("Running python worker on port 5000...")
worker_cmd = [
    os.path.join(os.path.dirname(__file__), "venv", "bin", "python"), "-u",
    "impala_worker.py", "--port", "5000", "--dataset_dir", "rl_test_dataset", "--test"
]
worker_proc = subprocess.Popen(worker_cmd)

try:
    worker_proc.wait(timeout=45)
    print(f"Worker exited with code: {worker_proc.returncode}")
except subprocess.TimeoutExpired:
    print("Worker ran for 45 seconds without exiting.")
    worker_proc.kill()
    worker_proc.wait()

print("Cleaning up Java server...")
server_proc.kill()
server_proc.wait()
