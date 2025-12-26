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


def run_command(cmd, description, env=None):
    """
    Run a shell command and return success status.
    
    Args:
        cmd: Command list to execute
        description: Description for logging
        env: Optional environment variables dict
    
    Returns:
        True if successful, False otherwise
    """
    print(f"  {description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {description} failed!")
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
    
    success = run_command(cmd, f"Running simulation with {routing_type}", env=env)
    
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
        f"*_ROUTER_TYPE.{routing_type}.pkl"
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
        """
    )
    
    parser.add_argument('--runs', '-n', type=int, required=True,
                       help='Number of simulation runs to perform')
    parser.add_argument('--params', '-p', type=str, default='topology_params.yaml',
                       help='Topology parameters YAML file (default: topology_params.yaml)')
    parser.add_argument('--routing', '-r', nargs='+',
                       default=['AODV', 'ZRP', 'MANAGED_FLOOD'],
                       help='Routing protocols to test (default: AODV ZRP MANAGED_FLOOD)')
    parser.add_argument('--output', '-o', type=str,
                       default=None,
                       help='Base output directory (default: batch_results_TIMESTAMP)')
    parser.add_argument('--start-seed', type=int, default=1000,
                       help='Starting seed value (default: 1000)')
    
    args = parser.parse_args()
    
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
    
    # Print header
    print("\n" + "=" * 70)
    print(f"BATCH SIMULATION: {args.runs} runs × {len(args.routing)} routing types")
    print("=" * 70)
    print(f"Output: {base_dir}")
    print(f"Routing: {', '.join(args.routing)}")
    print(f"Params: {params_file}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    runs_completed = 0
    
    # Run simulations
    for run_id in range(1, args.runs + 1):
        print(f"\n{'=' * 70}")
        print(f"RUN {run_id}/{args.runs}")
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
            args.routing, datetime.now().isoformat()
        )
        
        # Run simulation for each routing type
        all_success = True
        for routing_type in args.routing:
            routing_dir = run_dir / routing_type
            success = run_simulation(config_file, routing_type, routing_dir)
            
            if not success:
                all_success = False
                print(f"  [FAIL] Simulation failed: {routing_type}")
        
        if all_success:
            runs_completed += 1
            print(f"\n  [OK] Run {run_id} completed successfully")
        else:
            print(f"\n  [FAIL] Run {run_id} completed with errors")
    
    # Create summary
    print("\n" + "=" * 70)
    print(f"BATCH COMPLETED: {runs_completed}/{args.runs} successful runs")
    print("=" * 70)
    
    create_batch_summary(base_dir, runs_completed, args.runs, args.routing, start_time)
    
    print(f"\n[OK] All outputs saved to: {base_dir}")
    print(f"[OK] Summary: {base_dir / 'BATCH_SUMMARY.txt'}")
    print(f"[OK] Metadata: {base_dir / 'batch_summary.json'}\n")


if __name__ == '__main__':
    main()
