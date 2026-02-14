"""
Compare reliability metrics between different routing methods.

This script loads reliability CSV files from multiple routing types and creates
comprehensive comparison visualizations using bar graphs and other plots.

Usage:
    python compare_reliability.py --batch-dir batch_results_20260202_132004
    
    Or specify custom output directory:
    python compare_reliability.py --batch-dir batch_results_20260202_132004 --output-dir output/plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys


def load_reliability_csv(filepath):
    """Load a reliability CSV file and handle empty cells."""
    if not Path(filepath).exists():
        return None
    
    df = pd.read_csv(filepath)
    # Replace empty strings with NaN
    df = df.replace('', np.nan)
    return df


def discover_runs(batch_dir):
    """Discover all run directories in a batch results folder."""
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        return []
    
    run_dirs = sorted([d for d in batch_path.iterdir() if d.is_dir() and d.name.startswith('run_')])
    return run_dirs


def aggregate_reliability_data(run_dirs, filename_pattern):
    """
    Load reliability data from multiple runs and compute average.
    
    Args:
        run_dirs: list of Path objects pointing to run directories
        filename_pattern: pattern to match reliability files (e.g., 'sensor_reliability')
    
    Returns:
        dict of {routing_type: averaged_dataframe}
    """
    routing_data = {}  # {routing_type: [df1, df2, ...]}
    
    for run_dir in run_dirs:
        # Search recursively in subdirectories (routing type folders)
        for csv_file in run_dir.rglob(f"*{filename_pattern}*.csv"):
            filename = csv_file.name
            
            # Extract routing type from filename
            if "ROUTER_TYPE." in filename:
                routing_type = filename.split("ROUTER_TYPE.")[1].replace(".csv", "")
            else:
                continue
            
            df = load_reliability_csv(csv_file)
            if df is not None:
                if routing_type not in routing_data:
                    routing_data[routing_type] = []
                routing_data[routing_type].append(df)
    
    # Compute averages for each routing type
    averaged_data = {}
    for routing_type, dfs in routing_data.items():
        if len(dfs) == 0:
            continue
        
        # Average the dataframes
        if 'node_id' in dfs[0].columns:
            # For sensor or broadcast reliability (node-based)
            averaged_data[routing_type] = average_node_based_dfs(dfs)
        else:
            # For DM reliability matrix
            averaged_data[routing_type] = average_matrix_dfs(dfs)
    
    return averaged_data


def average_node_based_dfs(dfs):
    """
    Average node-based reliability dataframes from multiple runs.
    
    Args:
        dfs: list of dataframes with 'node_id' column
    
    Returns:
        averaged dataframe
    """
    if len(dfs) == 1:
        return dfs[0]
    
    # Get all unique node IDs
    all_nodes = set()
    for df in dfs:
        all_nodes.update(df['node_id'].dropna().astype(int).tolist())
    all_nodes = sorted(all_nodes)
    
    # Determine column name for reliability
    reliability_col = [c for c in dfs[0].columns if 'reliability' in c][0]
    
    # Aggregate data for each node
    averaged_rows = []
    for node_id in all_nodes:
        values = []
        for df in dfs:
            node_data = df[df['node_id'] == node_id]
            if len(node_data) > 0:
                val = node_data[reliability_col].values[0]
                if not pd.isna(val):
                    values.append(float(val))
        
        if values:
            avg_value = np.mean(values)
            averaged_rows.append({'node_id': node_id, reliability_col: avg_value})
        else:
            averaged_rows.append({'node_id': node_id, reliability_col: np.nan})
    
    return pd.DataFrame(averaged_rows)


def average_matrix_dfs(dfs):
    """
    Average matrix-based reliability dataframes from multiple runs.
    
    Args:
        dfs: list of dataframes (matrices)
    
    Returns:
        averaged dataframe
    """
    if len(dfs) == 1:
        return dfs[0]
    
    # Convert all to numeric
    numeric_dfs = []
    for df in dfs:
        numeric_df = df.copy()
        for col in numeric_df.columns:
            numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        numeric_dfs.append(numeric_df)
    
    # Average the matrices
    # Start with first matrix as template
    result = numeric_dfs[0].copy()
    
    # For each cell, average across all runs
    for idx in result.index:
        for col in result.columns:
            values = []
            for df in numeric_dfs:
                if idx in df.index and col in df.columns:
                    val = df.loc[idx, col]
                    if not pd.isna(val):
                        values.append(val)
            
            if values:
                result.loc[idx, col] = np.mean(values)
            else:
                result.loc[idx, col] = np.nan
    
    return result


def compare_sensor_reliability(data_dict, output_dir=".", n_runs=0):
    """
    Compare sensor packet reliability to destination 0 across routing methods.
    
    Args:
        data_dict: dict of {routing_name: dataframe} (already averaged)
        output_dir: directory to save output plots
        n_runs: number of runs averaged
    """
    print("\n" + "="*60)
    print("SENSOR PACKET RELIABILITY COMPARISON")
    if n_runs > 0:
        print(f"(Averaged across {n_runs} runs)")
    print("="*60)
    
    data = {}
    for routing_name, df in data_dict.items():
        if df is not None:
            data[routing_name] = df
            # Convert reliability column to numeric
            df['reliability_to_dest_0'] = pd.to_numeric(df['reliability_to_dest_0'], errors='coerce')
            print(f"\nLoaded {routing_name}: {len(df)} nodes")
            # Calculate statistics (ignoring NaN)
            valid_data = df['reliability_to_dest_0'].dropna()
            if len(valid_data) > 0:
                print(f"  Average reliability: {valid_data.mean():.4f} ({valid_data.mean()*100:.2f}%)")
                print(f"  Min reliability: {valid_data.min():.4f}")
                print(f"  Max reliability: {valid_data.max():.4f}")
                print(f"  Nodes with data: {len(valid_data)}/{len(df)}")
    
    if not data:
        print("No sensor reliability data found!")
        return
    
    # Create comparison plots
    routing_types = list(data.keys())
    n_types = len(routing_types)
    
    # Get all node IDs
    all_nodes = set()
    for df in data.values():
        all_nodes.update(df['node_id'].dropna().astype(int).tolist())
    all_nodes = sorted(all_nodes)
    
    # Plot 1: Grouped bar chart for all nodes
    fig, ax = plt.subplots(figsize=(max(14, len(all_nodes) * 0.8), 7))
    
    bar_width = 0.8 / n_types
    x = np.arange(len(all_nodes))
    colors = plt.cm.Set3.colors
    
    for i, (routing_name, df) in enumerate(data.items()):
        # Get reliability for each node (use NaN if missing)
        reliabilities = []
        for node_id in all_nodes:
            node_data = df[df['node_id'] == node_id]
            if len(node_data) > 0:
                rel = node_data['reliability_to_dest_0'].values[0]
                reliabilities.append(rel if not pd.isna(rel) else 0)
            else:
                reliabilities.append(0)
        
        offset = (i - n_types / 2) * bar_width + bar_width / 2
        bars = ax.bar(
            x + offset,
            reliabilities,
            bar_width,
            label=routing_name,
            color=colors[i % len(colors)],
            alpha=0.85,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add value labels on bars (only for non-zero values)
        for j, (bar, val) in enumerate(zip(bars, reliabilities)):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{val:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=7,
                    rotation=0
                )
    
    ax.set_xlabel('Node ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reliability to Destination 0', fontsize=12, fontweight='bold')
    title = 'Sensor Packet Reliability Comparison (All Nodes)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_nodes, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(title='Routing Method', loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = Path(output_dir) / "sensor_reliability_comparison_all_nodes.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()
    
    # Plot 2: Average reliability comparison (bar chart)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    routing_names = []
    avg_reliabilities = []
    std_reliabilities = []
    
    for routing_name, df in data.items():
        valid_data = df['reliability_to_dest_0'].dropna()
        if len(valid_data) > 0:
            routing_names.append(routing_name)
            avg_reliabilities.append(valid_data.mean())
            std_reliabilities.append(valid_data.std())
    
    bars = ax.bar(
        routing_names,
        avg_reliabilities,
        color=colors[:len(routing_names)],
        alpha=0.85,
        edgecolor='black',
        linewidth=1.5,
        yerr=std_reliabilities,
        capsize=10
    )
    
    # Add value labels
    for bar, avg, std in zip(bars, avg_reliabilities, std_reliabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02,
            f'{avg:.4f}\n({avg*100:.2f}%)',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.set_ylabel('Average Reliability', fontsize=12, fontweight='bold')
    title = 'Average Sensor Packet Reliability Comparison'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = Path(output_dir) / "sensor_reliability_average.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # Plot 3: Only nodes with data from both routing methods
    common_nodes = None
    for df in data.values():
        valid_nodes = set(df[df['reliability_to_dest_0'].notna()]['node_id'].astype(int))
        if common_nodes is None:
            common_nodes = valid_nodes
        else:
            common_nodes = common_nodes.intersection(valid_nodes)
    
    if common_nodes and len(common_nodes) > 0:
        common_nodes = sorted(common_nodes)
        
        fig, ax = plt.subplots(figsize=(max(10, len(common_nodes) * 0.6), 6))
        
        bar_width = 0.8 / n_types
        x = np.arange(len(common_nodes))
        
        for i, (routing_name, df) in enumerate(data.items()):
            reliabilities = []
            for node_id in common_nodes:
                rel = df[df['node_id'] == node_id]['reliability_to_dest_0'].values[0]
                reliabilities.append(rel)
            
            offset = (i - n_types / 2) * bar_width + bar_width / 2
            bars = ax.bar(
                x + offset,
                reliabilities,
                bar_width,
                label=routing_name,
                color=colors[i % len(colors)],
                alpha=0.85,
                edgecolor='black',
                linewidth=0.5
            )
            
            # Add value labels
            for bar, val in zip(bars, reliabilities):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f'{val:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        ax.set_xlabel('Node ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('Reliability to Destination 0', fontsize=12, fontweight='bold')
        title = f'Sensor Reliability Comparison (Common Nodes Only: {len(common_nodes)} nodes)'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(common_nodes, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend(title='Routing Method', loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_file = Path(output_dir) / "sensor_reliability_common_nodes.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


def compare_dm_reliability(data_dict, output_dir=".", n_runs=0):
    """
    Compare DM packet reliability matrices across routing methods.
    
    Args:
        data_dict: dict of {routing_name: dataframe} (already averaged)
        output_dir: directory to save output plots
        n_runs: number of runs averaged
    """
    print("\n" + "="*60)
    print("DM PACKET RELIABILITY COMPARISON")
    if n_runs > 0:
        print(f"(Averaged across {n_runs} runs)")
    print("="*60)
    
    data = {}
    for routing_name, df in data_dict.items():
        if df is not None:
            # Convert all columns to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            data[routing_name] = df
        
        # Calculate statistics
        values = df.values.flatten()
        valid_values = values[~np.isnan(values)]
        
        print(f"\nLoaded {routing_name}: {df.shape[0]}x{df.shape[1]} matrix")
        if len(valid_values) > 0:
            print(f"  Average reliability: {valid_values.mean():.4f} ({valid_values.mean()*100:.2f}%)")
            print(f"  Min reliability: {valid_values.min():.4f}")
            print(f"  Max reliability: {valid_values.max():.4f}")
            print(f"  Valid entries: {len(valid_values)}/{len(values)}")
    
    if not data:
        print("No DM reliability data found!")
        return
    
    # Plot: Average reliability per routing method
    fig, ax = plt.subplots(figsize=(8, 6))
    
    routing_names = []
    avg_reliabilities = []
    std_reliabilities = []
    
    colors = plt.cm.Set3.colors
    
    for routing_name, df in data.items():
        values = df.values.flatten()
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            routing_names.append(routing_name)
            avg_reliabilities.append(valid_values.mean())
            std_reliabilities.append(valid_values.std())
    
    bars = ax.bar(
        routing_names,
        avg_reliabilities,
        color=colors[:len(routing_names)],
        alpha=0.85,
        edgecolor='black',
        linewidth=1.5,
        yerr=std_reliabilities,
        capsize=10
    )
    
    # Add value labels
    for bar, avg, std in zip(bars, avg_reliabilities, std_reliabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.02,
            f'{avg:.4f}\n({avg*100:.2f}%)',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.set_ylabel('Average DM Reliability', fontsize=12, fontweight='bold')
    title = 'Average DM Packet Reliability Comparison'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = Path(output_dir) / "dm_reliability_average.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()


def compare_broadcast_reliability(data_dict, output_dir=".", n_runs=0):
    """
    Compare broadcast packet reliability across routing methods.
    
    Args:
        data_dict: dict of {routing_name: dataframe} (already averaged)
        output_dir: directory to save output plots
        n_runs: number of runs averaged
    """
    print("\n" + "="*60)
    print("BROADCAST PACKET RELIABILITY COMPARISON")
    if n_runs > 0:
        print(f"(Averaged across {n_runs} runs)")
    print("="*60)
    
    data = {}
    for routing_name, df in data_dict.items():
        if df is not None:
            data[routing_name] = df
            # Convert reliability column to numeric
            rel_col = [c for c in df.columns if 'reliability' in c][0]
            df[rel_col] = pd.to_numeric(df[rel_col], errors='coerce')
            print(f"\nLoaded {routing_name}: {len(df)} nodes")
            # Calculate statistics
            valid_data = df[rel_col].dropna()
            if len(valid_data) > 0:
                print(f"  Average reliability: {valid_data.mean():.4f} ({valid_data.mean()*100:.2f}%)")
    
    if not data:
        print("No broadcast reliability data found!")
        return
    
    # Plot: Average reliability comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    routing_names = []
    avg_reliabilities = []
    
    colors = plt.cm.Set3.colors
    
    for routing_name, df in data.items():
        rel_col = [c for c in df.columns if 'reliability' in c][0]
        valid_data = df[rel_col].dropna()
        if len(valid_data) > 0:
            routing_names.append(routing_name)
            avg_reliabilities.append(valid_data.mean())
    
    bars = ax.bar(
        routing_names,
        avg_reliabilities,
        color=colors[:len(routing_names)],
        alpha=0.85,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Add value labels
    for bar, avg in zip(bars, avg_reliabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{avg:.4f}\n({avg*100:.2f}%)',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    ax.set_ylabel('Average Broadcast Reliability', fontsize=12, fontweight='bold')
    title = 'Average Broadcast Packet Reliability Comparison'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = Path(output_dir) / "broadcast_reliability_average.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare reliability metrics between routing methods (averaged across multiple runs)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--batch-dir",
        required=True,
        help="Batch results directory containing run_001, run_002, etc. subdirectories"
    )
    parser.add_argument(
        "--output-dir",
        default="output/plots",
        help="Directory to save comparison plots (default: output/plots)"
    )
    
    args = parser.parse_args()
    
    batch_dir = Path(args.batch_dir)
    if not batch_dir.exists():
        print(f"Error: Batch directory '{batch_dir}' does not exist!")
        sys.exit(1)
    
    # Discover all run directories
    run_dirs = discover_runs(batch_dir)
    if not run_dirs:
        print(f"Error: No run directories (run_001, run_002, etc.) found in '{batch_dir}'!")
        sys.exit(1)
    
    print("="*60)
    print("RELIABILITY COMPARISON TOOL")
    print("="*60)
    print(f"\nBatch directory: {batch_dir.resolve()}")
    print(f"Found {len(run_dirs)} run(s):")
    for run_dir in run_dirs:
        print(f"  - {run_dir.name}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")
    
    # Aggregate reliability data from all runs
    print("\nAggregating data from all runs...")
    sensor_data = aggregate_reliability_data(run_dirs, "sensor_reliability")
    dm_data = aggregate_reliability_data(run_dirs, "dm_reliability_matrix")
    broadcast_data = aggregate_reliability_data(run_dirs, "broadcast_reliability")
    
    if not sensor_data and not dm_data and not broadcast_data:
        print("\nError: No reliability CSV files found in run directories!")
        print("\nExpected file patterns in each run directory:")
        print("  - sensor_reliability_to_dest0_ROUTER_TYPE.<routing>.csv")
        print("  - dm_reliability_matrix_ROUTER_TYPE.<routing>.csv")
        print("  - broadcast_reliability_ROUTER_TYPE.<routing>.csv")
        sys.exit(1)
    
    # Compare each metric type
    if sensor_data:
        print(f"\nFound sensor reliability data for routing types:")
        for routing in sensor_data.keys():
            print(f"  - {routing}")
        compare_sensor_reliability(sensor_data, args.output_dir, len(run_dirs))
    
    if dm_data:
        print(f"\nFound DM reliability data for routing types:")
        for routing in dm_data.keys():
            print(f"  - {routing}")
        compare_dm_reliability(dm_data, args.output_dir, len(run_dirs))
    
    if broadcast_data:
        print(f"\nFound broadcast reliability data for routing types:")
        for routing in broadcast_data.keys():
            print(f"  - {routing}")
        compare_broadcast_reliability(broadcast_data, args.output_dir, len(run_dirs))
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print(f"\nAll plots saved to: {Path(args.output_dir).resolve()}")
    print(f"Results averaged across {len(run_dirs)} run(s)")


if __name__ == "__main__":
    main()
