import os
import time
import argparse
import subprocess
import glob
import random
import threading
import sys
import signal

from logger_config import setup_logger

logger = setup_logger("Orchestrator")

# Set up relative bounds for meks
MEKFILES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mm-data", "data", "mekfiles", "meks"))
java_home = os.environ.get("JAVA_HOME", "/home/stuart_hatzioannou/jdk-21.0.2")

def get_random_meks(all_meks, num=1):
    if not all_meks:
        return ""
    return ",".join(random.choices(all_meks, k=num))

def stream_reader(stream, port_logger):
    """Reads lines from a subprocess stream and logs them."""
    for line in iter(stream.readline, b''):
        line_str = line.decode('utf-8', errors='replace').rstrip()
        if line_str:
            port_logger.info(line_str)
    stream.close()

def megamek_runner(port, mode, all_meks, shutdown_event, scenario=None, scenario_dir=None, options=None, max_meks=4):
    """Continuously runs the MegaMek server on the specified port until shutdown."""
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "megamek"))
    env = os.environ.copy()
    env["JAVA_HOME"] = java_home
    env["PATH"] = f"{os.path.join(java_home, 'bin')}:{env.get('PATH', '')}"
    env["MEGAMEK_OPTS"] = f"-DlogPath=logs/server_{port}"
    env["RL_SERVER_PORT"] = str(port)
    
    if options is None:
        options = []
    
    while not shutdown_event.is_set():
        cmd = [
            "build/install/MegaMek/bin/megamek", 
            "-rlexport"
        ]
        
        # Scenario selection
        current_scenario = scenario
        if scenario_dir and os.path.isdir(scenario_dir):
            scenarios = glob.glob(os.path.join(scenario_dir, "*.mms"))
            if scenarios:
                current_scenario = random.choice(scenarios)
        
        if current_scenario:
            cmd.extend(["-scenario", current_scenario])
        else:
            p1_num = random.randint(1, max_meks)
            p2_num = random.randint(1, max_meks)
            p1_meks = get_random_meks(all_meks, num=p1_num)
            p2_meks = get_random_meks(all_meks, num=p2_num)
            cmd.extend([
                "-randomMap", 
                "-p1meks", p1_meks, 
                "-p2meks", p2_meks
            ])
            
        if mode == "selfplay":
            cmd.append("-selfplay")
        else:
            cmd.append("-autogen")
            
        for opt in options:
            cmd.append(opt)
            
        port_logger = setup_logger(f"MegaMek {port}")
        port_logger.info("Starting match...")
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        # Start a thread to read Java stdout
        t_reader = threading.Thread(target=stream_reader, args=(proc.stdout, port_logger), daemon=True)
        t_reader.start()
        
        # Wait for the process to finish or shutdown to be requested
        while proc.poll() is None:
            if shutdown_event.is_set():
                proc.terminate()
                proc.wait()
                return
            time.sleep(1)

def python_worker_runner(port, dataset_dir, shutdown_event, device="cpu"):
    """Continuously runs the ImpalaWorker process."""
    cmd = [
        os.path.join(os.path.dirname(__file__), "venv", "bin", "python"), 
        "impala_worker.py", 
        "--port", str(port), 
        "--dataset_dir", dataset_dir,
        "--device", device
    ]
    
    cwd = os.path.dirname(__file__)
    
    while not shutdown_event.is_set():
        # Impala worker handles its own logging so we don't need to pipe it
        logger.info(f"Starting Worker {port}...")
        proc = subprocess.Popen(cmd, cwd=cwd)
        
        while proc.poll() is None:
            if shutdown_event.is_set():
                proc.terminate()
                proc.wait()
                return
            time.sleep(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-instances", type=int, default=4, help="Number of concurrent MegaMek servers")
    parser.add_argument("--mode", type=str, choices=["selfplay", "princess"], default="selfplay", help="Opponent mode")
    parser.add_argument("--base-port", type=int, default=4000)
    parser.add_argument("--scenario", type=str, default="", help="Path to a single .mms scenario file")
    parser.add_argument("--scenario-dir", type=str, default="", help="Directory containing .mms scenarios to randomly sample from")
    parser.add_argument("--max-meks", type=int, default=12, help="Maximum number of meks per side for random matches")
    parser.add_argument("--options", type=str, nargs="*", default=[], help="List of game options like -VICTORY_USE_KILL_COUNT=true")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run workers on (cpu, cuda:0, etc)")
    args = parser.parse_args()
    
    dataset_dir = "rl_sp_dataset" if args.mode == "selfplay" else "rl_princess_dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    logger.info("Pre-fetching all valid MTF Mek files...")
    all_mtf_files = glob.glob(os.path.join(MEKFILES_ROOT, "**", "*.mtf"), recursive=True)
    all_meks = [os.path.abspath(f) for f in all_mtf_files]
    logger.info(f"Loaded {len(all_meks)} available mechs.")
    
    shutdown_event = threading.Event()
    threads = []
    
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received! Terminating cluster...")
        shutdown_event.set()
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    for i in range(args.num_instances):
        server_port = args.base_port + (i * 10)
        worker1_port = server_port + 1000
        worker2_port = server_port + 1001
        
        # Start Server Thread
        t_server = threading.Thread(target=megamek_runner, args=(server_port, args.mode, all_meks, shutdown_event, args.scenario, args.scenario_dir, args.options, args.max_meks))
        t_server.start()
        threads.append(t_server)
        
        # Give server a slight head start to bind socket before python tries to connect
        time.sleep(2)
        
        # Start Python Worker Thread(s)
        t_w1 = threading.Thread(target=python_worker_runner, args=(worker1_port, dataset_dir, shutdown_event, args.device))
        t_w1.start()
        threads.append(t_w1)
        
        if args.mode == "selfplay":
            t_w2 = threading.Thread(target=python_worker_runner, args=(worker2_port, dataset_dir, shutdown_event, args.device))
            t_w2.start()
            threads.append(t_w2)
            
    print(f"\n==============================================")
    print(f" Orchestrator running {args.num_instances} matches in {args.mode} mode.")
    print(f" Dataset Directory: {dataset_dir}")
    print(f"==============================================\n")
    
    # Wait until shutdown
    while not shutdown_event.is_set():
        time.sleep(1)
        
    for t in threads:
        t.join()
        
    print("All processes terminated cleanly.")

if __name__ == "__main__":
    main()
