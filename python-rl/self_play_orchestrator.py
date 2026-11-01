import os
import time
import argparse
import subprocess
import glob
import random
import threading
import sys
import signal
import csv
import torch

from logger_config import setup_logger

logger = setup_logger("Orchestrator")

# Set up relative bounds for meks
MEKFILES_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mm-data", "data", "mekfiles", "meks"))
java_home = os.environ.get("JAVA_HOME", os.path.expanduser("~/jdk-21.0.2"))

def is_process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def terminate_process_group(proc, name="process", timeout=8.0):
    """Gracefully terminates a subprocess and its entire POSIX process group (e.g., bash wrapper + child JVM)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        logger.info(f"Sending SIGTERM to {name} process group (PGID {pgid}, PID {proc.pid})...")
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        try:
            proc.terminate()
        except OSError:
            return

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"{name} (PID {proc.pid}) did not exit within {timeout}s; sending SIGKILL to process group...")
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            pass


def cleanup_orphan_processes():
    """Finds and cleanly terminates any lingering headless MegaMek RL servers or IMPALA workers."""
    patterns = [
        ("IMPALA Worker", "impala_worker.py"),
        ("MegaMek RL Server", "megamek.MegaMek.*-rlexport"),
        ("MegaMek RL Offline", "megamek.MegaMek.*-rl_offline"),
        ("IMPALA Master", "impala_master.py"),
    ]
    for label, pat in patterns:
        try:
            res = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
            pids = [int(p) for p in res.stdout.strip().split() if p.strip().isdigit() and int(p) != os.getpid()]
            if pids:
                logger.info(f"Found {len(pids)} lingering {label} process(es): {pids}. Sending SIGTERM...")
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except OSError:
                        pass
                time.sleep(2.0)
                for pid in pids:
                    if is_process_alive(pid):
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except OSError:
                            pass
        except Exception as e:
            logger.warning(f"Error sweeping {label}: {e}")

def get_random_meks(all_meks, num=1):
    if not all_meks:
        return ""
    return ",".join(random.choices(all_meks, k=num))

def stream_reader(stream, port_logger, match_log_path=None):
    """Reads lines from a subprocess stream and logs them, optionally to a specific file."""
    f = None
    if match_log_path:
        f = open(match_log_path, "a", encoding="utf-8")
        
    for line in iter(stream.readline, b''):
        line_str = line.decode('utf-8', errors='replace').rstrip()
        if line_str:
            port_logger.info(line_str)
            if f:
                f.write(line_str + "\n")
                f.flush()
                
    if f:
        f.close()
    stream.close()

def megamek_runner(port, mode, all_meks, shutdown_event, scenario=None, scenario_dir=None, options=None, max_meks=4, use_bv_balancer=False, min_bv=3000, max_bv=8000, worker_id="default_worker", dataset_dir="data/rl_selfplay_trajectories", p_psi=0.25, midgame_save_dir="data/midgame_saves"):
    """Continuously runs the MegaMek server on the specified port until shutdown."""
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "megamek"))
    
    abs_log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), dataset_dir, worker_id, "logs"))
    os.makedirs(abs_log_dir, exist_ok=True)
    abs_midgame_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), midgame_save_dir))
    os.makedirs(abs_midgame_dir, exist_ok=True)
    
    env = os.environ.copy()
    env["JAVA_HOME"] = java_home
    env["PATH"] = f"{os.path.join(java_home, 'bin')}:{env.get('PATH', '')}"
    env["MEGAMEK_OPTS"] = f"-DlogPath={abs_log_dir}"
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
        
        # Check PSI (Perturbed State Initialization)
        midgame_saves = glob.glob(os.path.join(abs_midgame_dir, "*.sav")) + glob.glob(os.path.join(abs_midgame_dir, "*.sav.gz"))
        use_psi = (random.random() < p_psi) and (len(midgame_saves) > 0)
        
        if use_psi:
            loaded_save = random.choice(midgame_saves)
            cmd.extend(["-savegame", loaded_save])
        elif current_scenario:
            cmd.extend(["-scenario", current_scenario])
        elif use_bv_balancer:
            target_bv = random.randint(min_bv, max_bv)
            cmd.extend([
                "-randomMap",
                "-randomBV", str(target_bv),
                "-maxMeks", str(max_meks)
            ])
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
            
        # Enable midgame auto-saves
        cmd.extend(["-autoSaveMidGame", abs_midgame_dir])
            
        for opt in options:
            cmd.append(opt)
            
        try:
            match_id = int(time.time())
            match_log_file = os.path.join(abs_log_dir, f"megamek_{port}_{match_id}.log")
            
            port_logger = setup_logger(f"MegaMek {port}")
            port_logger.info(f"Starting match... logging to {match_log_file}")
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            
            # Start a thread to read Java stdout
            t_reader = threading.Thread(target=stream_reader, args=(proc.stdout, port_logger, match_log_file), daemon=True)
            t_reader.start()
            
            # Wait for the process to finish or shutdown to be requested
            while proc.poll() is None and is_process_alive(proc.pid):
                if shutdown_event.is_set():
                    terminate_process_group(proc, name=f"MegaMek Server {port}", timeout=6.0)
                    return
                time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to start MegaMek server on port {port}: {e}")
            time.sleep(5)

def python_worker_runner(port, dataset_dir, shutdown_event, device="cpu", worker_id="default_worker"):
    """Continuously runs the ImpalaWorker process."""
    cmd = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "bin", "python")), 
        "impala_worker.py", 
        "--port", str(port), 
        "--dataset_dir", dataset_dir,
        "--device", device,
        "--worker_id", worker_id,
    ]
    
    cwd = os.path.dirname(__file__)
    abs_log_dir = os.path.abspath(os.path.join(cwd, dataset_dir, worker_id, "logs"))
    os.makedirs(abs_log_dir, exist_ok=True)
    
    while not shutdown_event.is_set():
        try:
            # Impala worker handles its own logging so we don't need to pipe it
            worker_run_id = int(time.time())
            worker_log_file = os.path.join(abs_log_dir, f"worker_{port}_{worker_run_id}.log")
            
            logger.info(f"Starting Worker {port}... logging to {worker_log_file}")
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            
            worker_logger = setup_logger(f"Worker {port}")
            t_reader = threading.Thread(target=stream_reader, args=(proc.stdout, worker_logger, worker_log_file), daemon=True)
            t_reader.start()
            
            while proc.poll() is None and is_process_alive(proc.pid):
                if shutdown_event.is_set():
                    terminate_process_group(proc, name=f"Worker {port}", timeout=10.0)
                    return
                time.sleep(1)
        except Exception as e:
            logger.error(f"Failed to start Python worker on port {port}: {e}")
            time.sleep(5)

def master_runner(
    dataset_dir,
    device,
    force_bootstrap,
    shutdown_event,
    lambda_bc=1.0,
    lambda_bc_final=0.0,
    bc_warmup_steps=500,
    bc_decay_steps=5000,
    bc_schedule="cosine",
    lam_epistemic=0.5,
    lam_epistemic_final=0.0,
    epistemic_decay_steps=10000,
):
    """Runs the Impala Master trainer."""
    cmd = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "venv", "bin", "python")), 
        "impala_master.py",
        "--device", device,
        "--dataset-dir", dataset_dir,
        "--lambda-bc", str(lambda_bc),
        "--lambda-bc-final", str(lambda_bc_final),
        "--bc-warmup-steps", str(bc_warmup_steps),
        "--bc-decay-steps", str(bc_decay_steps),
        "--bc-schedule", str(bc_schedule),
        "--lam-epistemic", str(lam_epistemic),
        "--lam-epistemic-final", str(lam_epistemic_final),
        "--epistemic-decay-steps", str(epistemic_decay_steps),
    ]
    if force_bootstrap:
        cmd.append("--force-bootstrap")
    cwd = os.path.dirname(__file__)
    
    while not shutdown_event.is_set():
        try:
            logger.info("Starting Impala Master...")
            proc = subprocess.Popen(cmd, cwd=cwd, start_new_session=True)
            while proc.poll() is None and is_process_alive(proc.pid):
                if shutdown_event.is_set():
                    terminate_process_group(proc, name="Impala Master", timeout=15.0)
                    return
                time.sleep(1)
            if proc.poll() is not None and proc.poll() != 0 and not shutdown_event.is_set():
                logger.warning(f"Impala Master exited with code {proc.poll()}. Restarting in 5s...")
                time.sleep(5)
        except Exception as e:
            logger.error(f"Failed to start Impala Master: {e}")
            time.sleep(5)

def trajectory_monitor(dataset_dir, shutdown_event):
    """Monitors the dataset directory for new trajectories and logs metrics to CSV."""
    csv_path = os.path.join(dataset_dir, "metrics_log.csv")
    file_exists = os.path.exists(csv_path)
    processed_files = set()
    
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['timestamp', 'worker_id', 'win', 'total_reward'])
            
        while not shutdown_event.is_set():
            worker_dirs = glob.glob(os.path.join(dataset_dir, "*"))
            for wd in worker_dirs:
                if not os.path.isdir(wd) or os.path.basename(wd) == "logs": continue
                
                files = glob.glob(os.path.join(wd, "*.pt"))
                for f in files:
                    if f in processed_files:
                        continue
                        
                    try:
                        data = torch.load(f, map_location='cpu', weights_only=False)
                        if not data.get("is_done", False):
                            processed_files.add(f)
                            continue
                            
                        win = data.get("win", False)
                        total_reward = data.get("episode_reward", 0.0)
                        
                        # Extract timestamp from filename traj_{ep}_{timestamp}.pt
                        basename = os.path.basename(f)
                        parts = basename.replace(".pt", "").split("_")
                        timestamp = parts[-1] if len(parts) >= 3 else str(int(time.time()))
                        worker_id = os.path.basename(wd)
                        
                        writer.writerow([timestamp, worker_id, str(win), f"{total_reward:.4f}"])
                        csvfile.flush()
                        os.fsync(csvfile.fileno())
                        processed_files.add(f)
                    except Exception as e:
                        pass
            
            time.sleep(5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-instances", type=int, default=4, help="Number of concurrent MegaMek servers")
    parser.add_argument("--mode", type=str, choices=["selfplay", "princess"], default="princess", help="Opponent mode")
    parser.add_argument("--run-master", action="store_true", help="Spawn the Impala master node alongside the workers")
    parser.add_argument("--base-port", type=int, default=4000)
    parser.add_argument("--scenario", type=str, default="", help="Path to a single .mms scenario file")
    parser.add_argument("--scenario-dir", type=str, default="", help="Directory containing .mms scenarios to randomly sample from")
    parser.add_argument("--max-meks", type=int, default=12, help="Maximum number of meks per side for random matches")
    parser.add_argument("--no-bv-balancer", action="store_true", help="Disable MegaMek's force builder to create BV-balanced random forces")
    parser.add_argument("--min-bv", type=int, default=None, help="Minimum BV target for random forces")
    parser.add_argument("--max-bv", type=int, default=None, help="Maximum BV target for random forces")
    parser.add_argument("--options", type=str, nargs="*", default=[], help="List of game options like -VICTORY_USE_KILL_COUNT=true")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run workers on (cpu, cuda:0, etc)")
    parser.add_argument("--master-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run master learner on (cpu, cuda, etc)")
    parser.add_argument("--force-bootstrap", action="store_true", help="Force bootstrap the master from BC weights, overwriting latest RL weights")
    parser.add_argument("--p-psi", type=float, default=0.25, help="Probability of Perturbed State Initialization (loading mid-game save)")
    parser.add_argument("--midgame-save-dir", type=str, default="data/midgame_saves", help="Directory for mid-game save snapshots")
    parser.add_argument("--lambda-bc", type=float, default=1.0, help="Initial weight for supervised BC co-training loss")
    parser.add_argument("--lambda-bc-final", type=float, default=0.0, help="Final weight for supervised BC co-training loss")
    parser.add_argument("--bc-warmup-steps", type=int, default=500, help="Warmup steps before decaying lambda_bc")
    parser.add_argument("--bc-decay-steps", type=int, default=5000, help="Decay steps for lambda_bc")
    parser.add_argument("--bc-schedule", type=str, default="cosine", choices=["linear", "cosine", "exponential"], help="Annealing schedule for lambda_bc")
    parser.add_argument("--lam-epistemic", type=float, default=0.5, help="Initial weight for epistemic exploration bonus")
    parser.add_argument("--lam-epistemic-final", type=float, default=0.0, help="Final weight for epistemic exploration bonus")
    parser.add_argument("--epistemic-decay-steps", type=int, default=10000, help="Decay steps for lam_epistemic")
    parser.add_argument("--max-hours", type=float, default=0.0, help="Optional maximum runtime in hours before graceful shutdown (0 = unlimited)")
    parser.add_argument("--cleanup-orphans", action="store_true", help="Cleanly terminate any lingering MegaMek/IMPALA processes and exit")
    args = parser.parse_args()

    if args.cleanup_orphans:
        cleanup_orphan_processes()
        print("Orphan process sweep complete.")
        return
    
    if args.min_bv is None:
        if args.max_meks == 1:
            args.min_bv = 750
        elif args.max_meks <= 2:
            args.min_bv = 1500
        else:
            args.min_bv = 3000
            
    if args.max_bv is None:
        if args.max_meks == 1:
            args.max_bv = 2500
        elif args.max_meks <= 2:
            args.max_bv = 4500
        else:
            args.max_bv = 8000
            
    use_bv_balancer = not args.no_bv_balancer
    
    dataset_dir = "data/rl_selfplay_trajectories" if args.mode == "selfplay" else "data/rl_princess_trajectories"
    os.makedirs(dataset_dir, exist_ok=True)
    
    logger.info("Pre-fetching all valid MTF Mek files...")
    all_mtf_files = glob.glob(os.path.join(MEKFILES_ROOT, "**", "*.mtf"), recursive=True)
    all_meks = [os.path.abspath(f) for f in all_mtf_files]
    logger.info(f"Loaded {len(all_meks)} available mechs.")
    
    shutdown_requested = threading.Event()
    worker_shutdown_event = threading.Event()
    server_shutdown_event = threading.Event()
    master_shutdown_event = threading.Event()

    worker_threads = []
    server_threads = []
    master_threads = []

    # Start the trajectory monitor
    t_monitor = threading.Thread(target=trajectory_monitor, args=(dataset_dir, master_shutdown_event), daemon=True)
    t_monitor.start()

    if args.run_master:
        t_master = threading.Thread(
            target=master_runner,
            args=(
                dataset_dir,
                args.master_device,
                args.force_bootstrap,
                master_shutdown_event,
                args.lambda_bc,
                args.lambda_bc_final,
                args.bc_warmup_steps,
                args.bc_decay_steps,
                args.bc_schedule,
                args.lam_epistemic,
                args.lam_epistemic_final,
                args.epistemic_decay_steps,
            ),
        )
        t_master.start()
        master_threads.append(t_master)
        time.sleep(2)  # Give master a moment to initialize

    def signal_handler(sig, frame):
        if not shutdown_requested.is_set():
            logger.info(f"Shutdown signal ({sig}) received! Starting ordered 3-phase cluster shutdown...")
            shutdown_requested.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for i in range(args.num_instances):
        server_port = args.base_port + (i * 10)
        worker1_port = server_port + 1000
        worker2_port = server_port + 1001

        worker_id = f"worker_{server_port}"

        # Start Server Thread
        t_server = threading.Thread(
            target=megamek_runner,
            args=(
                server_port,
                args.mode,
                all_meks,
                server_shutdown_event,
                args.scenario,
                args.scenario_dir,
                args.options,
                args.max_meks,
                use_bv_balancer,
                args.min_bv,
                args.max_bv,
                worker_id,
                dataset_dir,
                args.p_psi,
                args.midgame_save_dir,
            ),
        )
        t_server.start()
        server_threads.append(t_server)

        # Give server a slight head start to bind socket before python tries to connect
        time.sleep(2)

        # Start Python Worker Thread(s)
        t_w1 = threading.Thread(
            target=python_worker_runner,
            args=(worker1_port, dataset_dir, worker_shutdown_event, args.device, worker_id),
        )
        t_w1.start()
        worker_threads.append(t_w1)

        if args.mode == "selfplay":
            t_w2 = threading.Thread(
                target=python_worker_runner,
                args=(worker2_port, dataset_dir, worker_shutdown_event, args.device, worker_id),
            )
            t_w2.start()
            worker_threads.append(t_w2)

    print(f"\n==============================================")
    print(f" Orchestrator running {args.num_instances} matches in {args.mode} mode.")
    print(f" Dataset Directory: {dataset_dir}")
    print(f"==============================================\n")

    start_time = time.time()
    while not shutdown_requested.is_set():
        if args.max_hours > 0 and (time.time() - start_time) >= args.max_hours * 3600.0:
            logger.info(f"Reached --max-hours ({args.max_hours}h). Initiating graceful shutdown...")
            shutdown_requested.set()
            break
        time.sleep(1)

    # Ordered 3-Phase Shutdown
    logger.info("[Shutdown Phase 1/3] Signaling IMPALA workers to flush in-flight trajectories...")
    worker_shutdown_event.set()
    for t in worker_threads:
        t.join(timeout=12.0)

    logger.info("[Shutdown Phase 2/3] Terminating headless MegaMek JVM process groups...")
    server_shutdown_event.set()
    for t in server_threads:
        t.join(timeout=10.0)

    logger.info("[Shutdown Phase 3/3] Signaling IMPALA Master to save final checkpoint and optimizer state...")
    master_shutdown_event.set()
    for t in master_threads:
        t.join(timeout=18.0)

    t_monitor.join(timeout=3.0)
    print("All processes terminated cleanly and state preserved.")

if __name__ == "__main__":
    main()
