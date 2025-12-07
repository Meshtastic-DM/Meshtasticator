"""
Compare reliability metrics between different routing methods.

This script loads reliability CSV files from multiple routing types and creates
comprehensive comparison visualizations using bar graphs and other plots.

Usage:
    python compare_reliability.py
    
    Or specify custom directory:
    python compare_reliability.py --input-dir output/
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


def compare_sensor_reliability(files_dict, output_dir="."):
    """
    Compare sensor packet reliability to destination 0 across routing methods.
    
    Args:
        files_dict: dict of {routing_name: filepath}
        output_dir: directory to save output plots
    """
    print("\n" + "="*60)
    print("SENSOR PACKET RELIABILITY COMPARISON")
    print("="*60)
    
    data = {}
    for routing_name, filepath in files_dict.items():
        df = load_reliability_csv(filepath)
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
    ax.set_title('Sensor Packet Reliability Comparison (All Nodes)', fontsize=14, fontweight='bold')
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
    ax.set_title('Average Sensor Packet Reliability Comparison', fontsize=14, fontweight='bold')
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
        ax.set_title(f'Sensor Reliability Comparison (Common Nodes Only: {len(common_nodes)} nodes)',
                     fontsize=13, fontweight='bold')
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


def compare_dm_reliability(files_dict, output_dir="."):
    """
    Compare DM packet reliability matrices across routing methods.
    
    Args:
        files_dict: dict of {routing_name: filepath}
        output_dir: directory to save output plots
    """
    print("\n" + "="*60)
    print("DM PACKET RELIABILITY COMPARISON")
    print("="*60)
    
    data = {}
    for routing_name, filepath in files_dict.items():
        if not Path(filepath).exists():
            continue
        
        df = pd.read_csv(filepath, index_col=0)
        df = df.replace('', np.nan)
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
    ax.set_title('Average DM Packet Reliability Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = Path(output_dir) / "dm_reliability_average.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()


def compare_broadcast_reliability(files_dict, output_dir="."):
    """
    Compare broadcast packet reliability across routing methods.
    
    Args:
        files_dict: dict of {routing_name: filepath}
        output_dir: directory to save output plots
    """
    print("\n" + "="*60)
    print("BROADCAST PACKET RELIABILITY COMPARISON")
    print("="*60)
    
    data = {}
    for routing_name, filepath in files_dict.items():
        df = load_reliability_csv(filepath)
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
    ax.set_title('Average Broadcast Packet Reliability Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_file = Path(output_dir) / "broadcast_reliability_average.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare reliability metrics between routing methods",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        default="output",
        help="Directory containing reliability CSV files (default: output)"
    )
    parser.add_argument(
        "--output-dir",
        default="output/plots",
        help="Directory to save comparison plots (default: output/plots)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        sys.exit(1)
    
    # Auto-detect reliability files
    sensor_files = {}
    dm_files = {}
    broadcast_files = {}
    
    for csv_file in input_dir.glob("*reliability*.csv"):
        filename = csv_file.name
        
        # Extract routing type from filename
        # Expected format: <type>_reliability_<metric>_ROUTER_TYPE.<routing>.csv
        if "ROUTER_TYPE." in filename:
            routing_type = filename.split("ROUTER_TYPE.")[1].replace(".csv", "")
        else:
            continue
        
        if "sensor_reliability" in filename:
            sensor_files[routing_type] = str(csv_file)
        elif "dm_reliability_matrix" in filename:
            dm_files[routing_type] = str(csv_file)
        elif "broadcast_reliability" in filename:
            broadcast_files[routing_type] = str(csv_file)
    
    if not sensor_files and not dm_files and not broadcast_files:
        print(f"No reliability CSV files found in '{input_dir}'!")
        print("\nExpected file patterns:")
        print("  - sensor_reliability_to_dest0_ROUTER_TYPE.<routing>.csv")
        print("  - dm_reliability_matrix_ROUTER_TYPE.<routing>.csv")
        print("  - broadcast_reliability_ROUTER_TYPE.<routing>.csv")
        sys.exit(1)
    
    print("="*60)
    print("RELIABILITY COMPARISON TOOL")
    print("="*60)
    print(f"\nInput directory: {input_dir.resolve()}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")
    
    # Compare each metric type
    if sensor_files:
        print(f"\nFound sensor reliability files:")
        for routing, path in sensor_files.items():
            print(f"  - {routing}: {Path(path).name}")
        compare_sensor_reliability(sensor_files, args.output_dir)
    
    if dm_files:
        print(f"\nFound DM reliability files:")
        for routing, path in dm_files.items():
            print(f"  - {routing}: {Path(path).name}")
        compare_dm_reliability(dm_files, args.output_dir)
    
    if broadcast_files:
        print(f"\nFound broadcast reliability files:")
        for routing, path in broadcast_files.items():
            print(f"  - {routing}: {Path(path).name}")
        compare_broadcast_reliability(broadcast_files, args.output_dir)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print(f"\nAll plots saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
