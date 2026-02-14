"""
Batch Simulation Runner for Multiple Topology Runs

This script automates running multiple simulations with different random topologies
and different routing protocols, organizing outputs in separate directories.

Usage:
    python run_batch_simulations.py --runs 10 --routing AODV ZRP MANAGED_FLOOD
    python run_batch_simulations.py --runs 5 --params topology_params.yaml --routing ZRP
"""

import argparse
import subprocess
import shutil
import time
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import pickle


#############################
# GOOGLE DRIVE SYNC SUPPORT
#############################

def is_colab_environment():
    """
    Detect if running in Google Colab environment.
    
    Returns:
        True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_drive_mounted(drive_path='/content/drive'):
    """
    Check if Google Drive is mounted.
    
    Args:
        drive_path: Path where Drive should be mounted
    
    Returns:
        True if Drive is mounted and accessible
    """
    drive_marker = Path(drive_path) / 'MyDrive'
    return drive_marker.exists() and drive_marker.is_dir()


def sync_to_google_drive(source_dir, drive_base_path, batch_name):
    """
    Sync batch results to Google Drive.
    
    Args:
        source_dir: Local directory to sync
        drive_base_path: Base path in Google Drive
        batch_name: Name of the batch directory
    
    Returns:
        True if successful, False otherwise
    """
    try:
        source = Path(source_dir)
        if not source.exists():
            print(f"    [SYNC] Warning: Source directory not found: {source}")
            return False
        
        # Create destination path
        drive_dest = Path(drive_base_path) / batch_name
        
        # Ensure parent directory exists
        drive_dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rsync-like behavior: sync only changed files
        if drive_dest.exists():
            # Update existing directory
            for item in source.rglob('*'):
                if item.is_file():
                    rel_path = item.relative_to(source)
                    dest_file = drive_dest / rel_path
                    
                    # Create parent directory if needed
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy if newer or doesn't exist
                    if not dest_file.exists() or item.stat().st_mtime > dest_file.stat().st_mtime:
                        shutil.copy2(str(item), str(dest_file))
        else:
            # First time: copy entire directory
            shutil.copytree(str(source), str(drive_dest))
        
        return True
    
    except Exception as e:
        print(f"    [SYNC] Error syncing to Google Drive: {e}")
        return False


#############################
# CHECKPOINT SUPPORT
#############################

def save_checkpoint(base_dir, completed_runs, total_runs, routing_types, start_time):
    """
    Save checkpoint information to allow resuming interrupted simulations.
    
    Args:
        base_dir: Base output directory
        completed_runs: List of successfully completed run IDs
        total_runs: Total number of runs to complete
        routing_types: List of routing protocols
        start_time: Start time of the batch
    """
    checkpoint = {
        "completed_runs": completed_runs,
        "total_runs": total_runs,
        "routing_types": routing_types,
        "start_time": start_time,
        "last_updated": time.time()
    }
    
    checkpoint_file = base_dir / ".checkpoint.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    # Also save as JSON for human readability
    checkpoint_json = base_dir / "checkpoint.json"
    checkpoint_readable = {
        "completed_runs": completed_runs,
        "total_runs": total_runs,
        "routing_types": routing_types,
        "start_time_iso": datetime.fromtimestamp(start_time).isoformat(),
        "last_updated_iso": datetime.fromtimestamp(checkpoint["last_updated"]).isoformat(),
        "progress_percentage": len(completed_runs) / total_runs * 100 if total_runs > 0 else 0
    }
    with open(checkpoint_json, 'w') as f:
        json.dump(checkpoint_readable, f, indent=2)


def load_checkpoint(base_dir):
    """
    Load checkpoint information from a previous simulation run.
    
    Args:
        base_dir: Base output directory
    
    Returns:
        Checkpoint dictionary or None if no checkpoint exists
    """
    checkpoint_file = base_dir / ".checkpoint.pkl"
    
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None


def get_remaining_runs(base_dir, total_runs):
    """
    Determine which runs still need to be completed.
    
    Args:
        base_dir: Base output directory
        total_runs: Total number of runs expected
    
    Returns:
        Set of run IDs that need to be completed
    """
    all_runs = set(range(1, total_runs + 1))
    completed = set()
    
    # Check which run directories exist and have metadata
    for run_id in all_runs:
        run_dir = base_dir / f"run_{run_id:03d}"
        metadata_file = run_dir / "run_metadata.json"
        
        if metadata_file.exists():
            completed.add(run_id)
    
    remaining = all_runs - completed
    return remaining, completed


def get_python_executable():
    """
    Get the appropriate Python executable path.
    Tries to detect and use virtual environment if available.
    
    Returns:
        Path to Python executable
    """
    # Check if we're already in a venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a venv, use current interpreter
        return sys.executable
    
    # Look for venv in common locations
    venv_locations = [
        Path('venv'),
        Path('.venv'),
        Path('env'),
        Path('.env')
    ]
    
    for venv_path in venv_locations:
        if venv_path.exists():
            # Try Windows path first
            venv_python = venv_path / 'Scripts' / 'python.exe'
            if venv_python.exists():
                print(f"[INFO] Using virtual environment: {venv_path}")
                return str(venv_python)
            
            # Try Unix/Linux path
            venv_python = venv_path / 'bin' / 'python'
            if venv_python.exists():
                print(f"[INFO] Using virtual environment: {venv_path}")
                return str(venv_python)
    
    # No venv found, use current Python
    return sys.executable


def run_command(cmd, description, env=None, log_file=None):
    """
    Run a shell command and return success status.
    Optionally write stdout/stderr to a log file.
    """
    print(f"  {description}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        if log_file is not None:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("STDOUT:\n")
                f.write(result.stdout or "")
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr or "")

        return True

    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {description} failed!")

        if log_file is not None:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("STDOUT:\n")
                f.write(e.stdout or "")
                f.write("\n\nSTDERR:\n")
                f.write(e.stderr or "")

        if e.stderr:
            print(f"  {e.stderr}")
        if e.stdout:
            print(f"  {e.stdout}")

        return False


def generate_topology(run_id, params_file, output_dir, seed):
    """
    Generate a random topology configuration.
    
    Args:
        run_id: Run identifier
        params_file: Path to topology parameters YAML
        output_dir: Directory to save config
        seed: Random seed
    
    Returns:
        Path to generated config file, or None on failure
    """
    config_file = output_dir / "config.yaml"
    
    python_exe = get_python_executable()
    cmd = [
        python_exe, "generate_topology.py",
        "--input-params", str(params_file),
        "--output", str(config_file),
        "--seed", str(seed),
        "--quiet"
    ]
    
    success = run_command(cmd, f"Generating topology (seed={seed})")
    
    return config_file if success else None


def run_simulation(config_file, routing_type, output_dir):
    """
    Run a simulation with specified routing type.
    
    Args:
        config_file: Path to topology config YAML
        routing_type: Routing protocol name
        output_dir: Directory to save outputs
    
    Returns:
        True if successful, False otherwise
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if config file exists
    if not config_file.exists():
        print(f"  ERROR: Config file not found: {config_file}")
        return False
    
    python_exe = get_python_executable()
    
    # Run simulation using --route-type argument
    cmd = [
        python_exe, "loraMesh.py",
        "--from-file", str(config_file),
        "--route-type", routing_type
    ]
    
    # Set environment to use non-interactive matplotlib backend
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'  # Use non-interactive backend
    
    log_file = output_dir / "terminal_output.txt"

    success = run_command(
        cmd,
        f"Running simulation with {routing_type}",
        env=env,
        log_file=log_file
    )

    
    if success:
        # Move output files to run-specific directory
        move_output_files(output_dir, routing_type)
    
    return success


def move_output_files(output_dir, routing_type):
    """
    Move simulation output CSV, PNG, and PKL files to the run-specific directory.
    
    Args:
        output_dir: Directory to move files to
        routing_type: Routing protocol name for file matching
    """
    output_source = Path("output")
    if not output_source.exists():
        return
    
    # Patterns to match output files for this routing type (CSV, PNG, and PKL)
    patterns = [
        f"*_{routing_type}.csv",
        f"*_ROUTER_TYPE.{routing_type}.csv",
        f"*_{routing_type}.png",
        f"*_ROUTER_TYPE.{routing_type}.png",
        f"*_{routing_type}.pkl",
        f"*_ROUTER_TYPE.{routing_type}.pkl",
        f"plots/*_ROUTER_TYPE.{routing_type}.png",
        f"battery_node_*.csv"
    ]
    
    files_moved = 0
    for pattern in patterns:
        for src_file in output_source.glob(pattern):
            dst_file = output_dir / src_file.name
            try:
                shutil.move(str(src_file), str(dst_file))
                files_moved += 1
            except Exception as e:
                print(f"    Warning: Could not move {src_file.name}: {e}")
    
    if files_moved > 0:
        print(f"    Moved {files_moved} output files")


def create_run_metadata(run_dir, run_id, params_file, seed, routing_types, timestamp):
    """
    Create a metadata JSON file for the run.
    
    Args:
        run_dir: Directory for this run
        run_id: Run identifier
        params_file: Path to topology params file
        seed: Random seed used
        routing_types: List of routing protocols tested
        timestamp: ISO timestamp of run start
    """
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "seed": seed,
        "params_file": str(params_file),
        "routing_types": routing_types,
        "config_file": str(run_dir / "config.yaml")
    }
    
    metadata_file = run_dir / "run_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def create_batch_summary(base_dir, runs_completed, total_runs, routing_types, start_time):
    """
    Create a summary report for the batch simulation.
    
    Args:
        base_dir: Base output directory
        runs_completed: Number of successful runs
        total_runs: Total number of runs attempted
        routing_types: List of routing protocols tested
        start_time: Start time of batch
    """
    duration = time.time() - start_time
    
    summary = {
        "batch_timestamp": datetime.now().isoformat(),
        "total_runs": total_runs,
        "successful_runs": runs_completed,
        "failed_runs": total_runs - runs_completed,
        "routing_types": routing_types,
        "duration_seconds": round(duration, 2),
        "duration_minutes": round(duration / 60, 2),
        "output_directory": str(base_dir)
    }
    
    # Save JSON summary
    summary_file = base_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create human-readable text summary
    report = []
    report.append("=" * 70)
    report.append("BATCH SIMULATION SUMMARY")
    report.append("=" * 70)
    report.append(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Duration: {duration / 60:.2f} minutes")
    report.append(f"\nRuns: {runs_completed}/{total_runs} successful")
    report.append(f"Routing Types: {', '.join(routing_types)}")
    report.append(f"\nOutput Directory: {base_dir}")
    report.append("\nDirectory Structure:")
    report.append(f"  {base_dir.name}/")
    report.append(f"    ├── batch_summary.json")
    report.append(f"    ├── run_001/")
    report.append(f"    │   ├── config.yaml")
    report.append(f"    │   ├── run_metadata.json")
    for routing in routing_types:
        report.append(f"    │   ├── {routing}/")
        report.append(f"    │   │   ├── sensor_packets_{routing}.csv")
        report.append(f"    │   │   ├── dm_packets_{routing}.csv")
        report.append(f"    │   │   ├── broadcast_packets_{routing}.csv")
        report.append(f"    │   │   ├── sensor_reliability_to_dest0_{routing}.csv")
        report.append(f"    │   │   ├── dm_reliability_matrix_{routing}.csv")
        report.append(f"    │   │   └── broadcast_reliability_{routing}.csv")
    report.append(f"    ├── run_002/")
    report.append(f"    │   └── ...")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    # Save text summary
    report_file = base_dir / "BATCH_SUMMARY.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Print summary
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple simulations with different topologies and routing protocols',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 simulations with all routing types
  python run_batch_simulations.py --runs 10
  
  # Run 5 simulations with specific routing types
  python run_batch_simulations.py --runs 5 --routing AODV ZRP
  
  # Use custom topology parameters
  python run_batch_simulations.py --runs 10 --params topology_params.yaml
  
  # Specify custom output directory
  python run_batch_simulations.py --runs 10 --output my_batch_results
  
  # Enable checkpointing (for long runs)
  python run_batch_simulations.py --runs 50 --checkpoint
  
  # Resume from checkpoint (no --runs needed)
  python run_batch_simulations.py --resume batch_results_20260214_120000
  
  # Resume with auto-sync to Google Drive (Colab)
  python run_batch_simulations.py --resume batch_results_20260214_120000 --auto-sync
        """
    )
    
    parser.add_argument('--runs', '-n', type=int, required=False, default=None,
                       help='Number of simulation runs to perform (required unless using --resume)')
    parser.add_argument('--params', '-p', type=str, default='topology_params.yaml',
                       help='Topology parameters YAML file (default: topology_params.yaml)')
    parser.add_argument('--routing', '-r', nargs='+',
                       default=['AODV', 'MANAGED_FLOOD','BL_A_AODV'], #removed ZRP for testing
                       help='Routing protocols to test (default: AODV ZRP MANAGED_FLOOD BL_A_AODV)')
    parser.add_argument('--output', '-o', type=str,
                       default=None,
                       help='Base output directory (default: batch_results_TIMESTAMP)')
    parser.add_argument('--start-seed', type=int, default=1000,
                       help='Starting seed value (default: 1000)')
    parser.add_argument('--checkpoint', action='store_true',
                       help='Enable checkpoint support to resume interrupted simulations')
    parser.add_argument('--resume', type=str, metavar='DIR',
                       help='Resume from a previous batch directory')
    parser.add_argument('--sync-to-drive', type=str, metavar='PATH',
                       help='Auto-sync results to Google Drive after each run (e.g., /content/drive/MyDrive/Meshtasticator_Results)')
    parser.add_argument('--auto-sync', action='store_true',
                       help='Auto-detect Colab and sync to Drive (uses /content/drive/MyDrive/Meshtasticator_Results)')
    
    args = parser.parse_args()
    
    # Validate --runs requirement
    if not args.resume and args.runs is None:
        parser.error("--runs is required when not using --resume")
    
    # Handle Google Drive sync
    drive_sync_path = None
    if args.auto_sync:
        if is_colab_environment():
            if is_drive_mounted():
                drive_sync_path = '/content/drive/MyDrive/Meshtasticator_Results'
                print("[DRIVE-SYNC] Auto-sync enabled to Google Drive")
            else:
                print("[DRIVE-SYNC] Warning: Colab detected but Drive not mounted")
                print("[DRIVE-SYNC] Run: from google.colab import drive; drive.mount('/content/drive')")
        else:
            print("[DRIVE-SYNC] Auto-sync requested but not in Colab environment")
    elif args.sync_to_drive:
        drive_sync_path = args.sync_to_drive
        if not Path(drive_sync_path).exists():
            print(f"[DRIVE-SYNC] Warning: Drive path not accessible: {drive_sync_path}")
            print(f"[DRIVE-SYNC] Will attempt to create it...")
            try:
                Path(drive_sync_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"[DRIVE-SYNC] Error: Could not create path: {e}")
                drive_sync_path = None
    
    # Handle resume mode
    if args.resume:
        base_dir = Path(args.resume)
        if not base_dir.exists():
            parser.error(f"Resume directory not found: {args.resume}")
        
        # Load checkpoint
        checkpoint = load_checkpoint(base_dir)
        if checkpoint is None:
            print(f"No checkpoint found in {base_dir}, starting fresh scan...")
            # Will scan for completed runs below
        else:
            print(f"\n✓ Loaded checkpoint from {base_dir}")
            print(f"✓ Previously completed: {len(checkpoint['completed_runs'])}/{checkpoint['total_runs']} runs")
        
        # Load parameters from the existing run
        params_file = base_dir / "topology_params_used.yaml"
        if not params_file.exists():
            parser.error(f"Cannot find topology_params_used.yaml in {args.resume}")
        
        # Load metadata from first run to get routing types
        for run_dir in sorted(base_dir.glob("run_*")):
            metadata_file = run_dir / "run_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    routing_types = metadata.get('routing_types', args.routing)
                    break
        else:
            routing_types = args.routing
        
        # Determine total runs from checkpoint or existing directories
        if checkpoint:
            total_runs = checkpoint['total_runs']
            start_time = checkpoint['start_time']
        else:
            # Count existing run directories
            run_dirs = list(base_dir.glob("run_*"))
            if run_dirs:
                # Get the highest run number from existing directories
                total_runs = max([int(d.name.split('_')[1]) for d in run_dirs])
            elif args.runs is not None:
                # Use provided runs if available
                total_runs = args.runs
            else:
                parser.error("Cannot determine total runs. No checkpoint found and no runs in directory.")
            start_time = time.time()
        
        print(f"✓ Resuming with {len(routing_types)} routing types: {', '.join(routing_types)}")
        args.checkpoint = True  # Enable checkpointing when resuming
        
    else:
        # Validate
        if args.runs < 1:
            parser.error("Number of runs must be at least 1")
        
        params_file = Path(args.params)
        if not params_file.exists():
            parser.error(f"Parameter file not found: {args.params}")
        
        # Create base output directory
        if args.output:
            base_dir = Path(args.output)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = Path(f"batch_results_{timestamp}")
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy params file to output directory for reference
        shutil.copy2(params_file, base_dir / f"topology_params_used.yaml")
        
        routing_types = args.routing
        total_runs = args.runs
        start_time = time.time()
    
    # Determine which runs need to be completed
    remaining_runs, completed_runs_set = get_remaining_runs(base_dir, total_runs)
    completed_runs_list = sorted(list(completed_runs_set))
    
    # Determine which runs need to be completed
    remaining_runs, completed_runs_set = get_remaining_runs(base_dir, total_runs)
    completed_runs_list = sorted(list(completed_runs_set))
    
    # Print header
    print("\n" + "=" * 70)
    if args.resume:
        print(f"RESUMING BATCH SIMULATION")
        print(f"Already completed: {len(completed_runs_list)}/{total_runs} runs")
        print(f"Remaining: {len(remaining_runs)} runs")
    else:
        print(f"BATCH SIMULATION: {total_runs} runs × {len(routing_types)} routing types")
    print("=" * 70)
    print(f"Output: {base_dir}")
    print(f"Routing: {', '.join(routing_types)}")
    print(f"Params: {params_file}")
    if args.checkpoint:
        print(f"Checkpointing: ENABLED ✓")
    if drive_sync_path:
        print(f"Drive Sync: ENABLED ✓ → {drive_sync_path}")
    print("=" * 70 + "\n")
    
    runs_completed = len(completed_runs_list)
    
    # Run simulations (only for remaining runs)
    runs_to_process = sorted(remaining_runs) if remaining_runs else []
    
    if not runs_to_process:
        print("\n✓ All runs already completed!")
    
    for run_id in runs_to_process:
        print(f"\n{'=' * 70}")
        print(f"RUN {run_id}/{total_runs} (Progress: {runs_completed}/{total_runs} completed)")
        print("=" * 70)
        
        # Create run directory
        run_dir = base_dir / f"run_{run_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate topology
        seed = args.start_seed + run_id
        config_file = generate_topology(run_id, params_file, run_dir, seed)
        
        if config_file is None:
            print(f"  [FAIL] Run {run_id} failed: Could not generate topology")
            continue
        
        # Create metadata
        create_run_metadata(
            run_dir, run_id, params_file, seed, 
            routing_types, datetime.now().isoformat()
        )
        
        # Run simulation for each routing type
        all_success = True
        for routing_type in routing_types:
            routing_dir = run_dir / routing_type
            success = run_simulation(config_file, routing_type, routing_dir)
            
            if not success:
                all_success = False
                print(f"  [FAIL] Simulation failed: {routing_type}")
        
        if all_success:
            runs_completed += 1
            completed_runs_list.append(run_id)
            print(f"\n  [OK] Run {run_id} completed successfully")
            
            # Save checkpoint after each successful run
            if args.checkpoint:
                save_checkpoint(base_dir, completed_runs_list, total_runs, routing_types, start_time)
                print(f"  [CHECKPOINT] Progress saved ({runs_completed}/{total_runs})")
            
            # Sync to Google Drive after each successful run
            if drive_sync_path:
                print(f"  [SYNC] Syncing to Google Drive...")
                sync_success = sync_to_google_drive(base_dir, drive_sync_path, base_dir.name)
                if sync_success:
                    print(f"  [SYNC] ✓ Backed up to Drive: {Path(drive_sync_path) / base_dir.name}")
                else:
                    print(f"  [SYNC] ✗ Sync failed (results still saved locally)")
        else:
            print(f"\n  [FAIL] Run {run_id} completed with errors")
    
    # Create summary
    print("\n" + "=" * 70)
    print(f"BATCH COMPLETED: {runs_completed}/{total_runs} successful runs")
    print("=" * 70)
    
    create_batch_summary(base_dir, runs_completed, total_runs, routing_types, start_time)
    
    print(f"\n[OK] All outputs saved to: {base_dir}")
    print(f"[OK] Summary: {base_dir / 'BATCH_SUMMARY.txt'}")
    print(f"[OK] Metadata: {base_dir / 'batch_summary.json'}")
    
    if args.checkpoint:
        print(f"[OK] Checkpoint: {base_dir / 'checkpoint.json'}")
    
    if drive_sync_path:
        print(f"[OK] Google Drive: {Path(drive_sync_path) / base_dir.name}")
        print(f"     All results continuously backed up to Drive ✓")
    
    print()


if __name__ == '__main__':
    main()
