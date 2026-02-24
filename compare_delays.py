"""
Compare packet delays between different routing methods.

This script loads delay CSV files from multiple routing types and creates
comprehensive delay comparison visualizations using box plots, violin plots, and bar charts.

Usage:
    python compare_delays.py --batch-dir batch_results_20260202_132004
    
    Or specify custom output directory:
    python compare_delays.py --batch-dir batch_results_20260202_132004 --output-dir output/plots
"""

import pandas as pd
import numpy as np
import os

# Configure matplotlib backend for headless environments (Colab, servers)
import matplotlib
backend = os.environ.get('MPLBACKEND', 'Agg')  # Default to Agg for compatibility
if backend == 'TkAgg':
    try:
        matplotlib.use("TkAgg")
    except ImportError:
        print('Warning: TkAgg not available. Using Agg backend (non-interactive).')
        matplotlib.use('Agg')
else:
    matplotlib.use(backend)

import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys


def load_delay_csv(filepath):
    """Load a delay CSV file."""
    if not Path(filepath).exists():
        return None
    
    df = pd.read_csv(filepath)
    # Ensure delay_value is numeric
    df['delay_value'] = pd.to_numeric(df['delay_value'], errors='coerce')
    df = df.dropna(subset=['delay_value'])
    return df


def discover_runs(batch_dir):
    """Discover all run directories in a batch results folder."""
    batch_path = Path(batch_dir)
    if not batch_path.exists():
        return []
    
    run_dirs = sorted([d for d in batch_path.iterdir() if d.is_dir() and d.name.startswith('run_')])
    return run_dirs


def aggregate_delay_data(run_dirs, filename_pattern):
    """
    Load delay data from multiple runs and aggregate.
    
    Args:
        run_dirs: list of Path objects pointing to run directories
        filename_pattern: pattern to match delay files (e.g., 'sensor_packets')
    
    Returns:
        dict of {routing_type: concatenated_dataframe}
    """
    routing_data = {}  # {routing_type: [df1, df2, ...]}
    
    for run_dir in run_dirs:
        # Search recursively in subdirectories (routing type folders)
        for csv_file in run_dir.rglob(f"*{filename_pattern}*.csv"):
            filename = csv_file.name
            
            # Skip reliability files
            if "reliability" in filename:
                continue
            
            # Extract routing type from parent directory name (most reliable)
            # Files are organized as: run_XXX/ROUTING_TYPE/file_ROUTING_TYPE.csv
            parent_dir = csv_file.parent.name
            
            # If parent is a run directory, skip (file is in wrong location)
            if parent_dir.startswith('run_'):
                continue
            
            # Use parent directory as routing type
            routing_type = parent_dir
            
            df = load_delay_csv(csv_file)
            if df is not None:
                if routing_type not in routing_data:
                    routing_data[routing_type] = []
                routing_data[routing_type].append(df)
    
    # Concatenate all dataframes for each routing type
    aggregated_data = {}
    for routing_type, dfs in routing_data.items():
        if len(dfs) > 0:
            # Concatenate all runs together
            aggregated_data[routing_type] = pd.concat(dfs, ignore_index=True)
    
    return aggregated_data


def compare_delays_overview(data_dict, packet_type, output_dir=".", num_runs=1):
    """
    Create overview comparison of delays across routing methods.
    
    Args:
        data_dict: dict of {routing_name: dataframe}
        packet_type: 'sensor', 'dm', or 'broadcast'
        output_dir: directory to save output plots
        num_runs: number of runs aggregated
    """
    print(f"\n{'='*60}")
    print(f"{packet_type.upper()} PACKET DELAY COMPARISON")
    print(f"(Aggregated across {num_runs} runs)")
    print(f"{'='*60}")
    
    data = data_dict
    for routing_name, df in data_dict.items():
        if df is not None and len(df) > 0:
            delays = df['delay_value']
            print(f"\nLoaded {routing_name}: {len(df)} packets")
            print(f"  Mean delay: {delays.mean():.2f} ms")
            print(f"  Median delay: {delays.median():.2f} ms")
            print(f"  Min delay: {delays.min():.2f} ms")
            print(f"  Max delay: {delays.max():.2f} ms")
            print(f"  Std dev: {delays.std():.2f} ms")
    
    if not data:
        print(f"No {packet_type} delay data found!")
        return
    
    routing_types = list(data.keys())
    colors = plt.cm.Set3.colors
    
    # Plot 1: Box plot comparison
    fig, ax = plt.subplots(figsize=(10, 7))
    
    delay_lists = [data[rt]['delay_value'].values for rt in routing_types]
    
    bp = ax.boxplot(
        delay_lists,
        labels=routing_types,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.6
    )
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Customize box plot elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    plt.setp(bp['medians'], color='red', linewidth=2)
    plt.setp(bp['means'], color='blue', linewidth=2)
    
    ax.set_ylabel('Delay (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Routing Method', fontsize=12, fontweight='bold')
    ax.set_title(f'{packet_type.title()} Packet Delay Distribution Comparison', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    output_file = Path(output_dir) / f"{packet_type}_delay_boxplot.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()
    
    # Plot 2: Violin plot comparison
    fig, ax = plt.subplots(figsize=(10, 7))
    
    positions = np.arange(1, len(routing_types) + 1)
    parts = ax.violinplot(
        delay_lists,
        positions=positions,
        showmeans=True,
        showmedians=True,
        widths=0.7
    )
    
    # Color the violin plots
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(routing_types)
    ax.set_ylabel('Delay (ms)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Routing Method', fontsize=12, fontweight='bold')
    ax.set_title(f'{packet_type.title()} Packet Delay Distribution (Violin Plot)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_file = Path(output_dir) / f"{packet_type}_delay_violin.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # Plot 3: Mean and median comparison (bar chart)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    means = [data[rt]['delay_value'].mean() for rt in routing_types]
    medians = [data[rt]['delay_value'].median() for rt in routing_types]
    stds = [data[rt]['delay_value'].std() for rt in routing_types]
    
    # Mean delays
    bars1 = ax1.bar(routing_types, means, color=colors[:len(routing_types)], 
                     alpha=0.85, edgecolor='black', linewidth=1.5,
                     yerr=stds, capsize=10)
    
    for bar, mean, std in zip(bars1, means, stds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + max(means)*0.02,
                f'{mean:.1f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Mean Delay (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Packet Delay', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Median delays
    bars2 = ax2.bar(routing_types, medians, color=colors[:len(routing_types)], 
                     alpha=0.85, edgecolor='black', linewidth=1.5)
    
    for bar, median in zip(bars2, medians):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(medians)*0.02,
                f'{median:.1f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Median Delay (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Median Packet Delay', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle(f'{packet_type.title()} Packet Delay Statistics Comparison', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_file = Path(output_dir) / f"{packet_type}_delay_statistics.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # Plot 4: CDF (Cumulative Distribution Function) comparison
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for i, (routing_name, df) in enumerate(data.items()):
        delays = np.sort(df['delay_value'].values)
        cdf = np.arange(1, len(delays) + 1) / len(delays)
        ax.plot(delays, cdf, label=routing_name, linewidth=2.5, 
                color=colors[i % len(colors)], alpha=0.8)
    
    ax.set_xlabel('Delay (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'{packet_type.title()} Packet Delay CDF Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(title='Routing Method', fontsize=11, loc='lower right')
    
    # Add percentile lines
    for percentile in [0.5, 0.9, 0.95, 0.99]:
        ax.axhline(y=percentile, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(ax.get_xlim()[0] * 1.1, percentile, f'{int(percentile*100)}%', 
                fontsize=9, va='center', color='gray')
    
    plt.tight_layout()
    output_file = Path(output_dir) / f"{packet_type}_delay_cdf.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()
    
    # Plot 5: Percentile comparison table (as bar chart)
    percentiles = [50, 75, 90, 95, 99]
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(percentiles))
    width = 0.8 / len(routing_types)
    
    for i, (routing_name, df) in enumerate(data.items()):
        delays = df['delay_value'].values
        perc_values = [np.percentile(delays, p) for p in percentiles]
        
        offset = (i - len(routing_types) / 2) * width + width / 2
        bars = ax.bar(x + offset, perc_values, width, 
                     label=routing_name, 
                     color=colors[i % len(colors)],
                     alpha=0.85, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, perc_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delay (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'{packet_type.title()} Packet Delay Percentiles Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}th' for p in percentiles])
    ax.legend(title='Routing Method', loc='best')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    
    plt.tight_layout()
    output_file = Path(output_dir) / f"{packet_type}_delay_percentiles.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def create_summary_comparison(sensor_data, dm_data, broadcast_data, output_dir="."):
    """Create a single summary plot comparing all packet types and routing methods."""
    print(f"\n{'='*60}")
    print("OVERALL DELAY SUMMARY")
    print(f"{'='*60}")
    
    all_data = {}
    
    # Load all data
    for packet_type, data_dict in [('Sensor', sensor_data), ('DM', dm_data), ('Broadcast', broadcast_data)]:
        if not data_dict:
            continue
        for routing_name, df in data_dict.items():
            if df is not None and len(df) > 0:
                key = f"{packet_type}\n{routing_name}"
                all_data[key] = df['delay_value'].values
    
    if not all_data:
        print("No data available for summary!")
        return
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    labels = list(all_data.keys())
    delay_lists = [all_data[k] for k in labels]
    
    bp = ax.boxplot(
        delay_lists,
        labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.6
    )
    
    # Color by packet type
    colors_map = {'Sensor': plt.cm.Set3.colors[0], 'DM': plt.cm.Set3.colors[1], 
                  'Broadcast': plt.cm.Set3.colors[2]}
    
    for patch, label in zip(bp['boxes'], labels):
        packet_type = label.split('\n')[0]
        patch.set_facecolor(colors_map.get(packet_type, 'skyblue'))
        patch.set_alpha(0.7)
    
    plt.setp(bp['medians'], color='red', linewidth=2)
    plt.setp(bp['means'], color='blue', linewidth=2)
    
    ax.set_ylabel('Delay (ms)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Packet Type & Routing Method', fontsize=13, fontweight='bold')
    ax.set_title('Overall Packet Delay Comparison (All Types & Routing Methods)', 
                 fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_yscale('log')
    plt.xticks(rotation=45, ha='right')
    
    # Add legend for median/mean lines
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    output_file = Path(output_dir) / "overall_delay_comparison.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {output_file}")
    plt.close()
    
    # Print summary statistics table
    print("\n" + "="*60)
    print("SUMMARY STATISTICS TABLE")
    print("="*60)
    print(f"{'Packet Type & Routing':<30} {'Mean':<12} {'Median':<12} {'P95':<12}")
    print("-" * 60)
    for label in labels:
        delays = all_data[label]
        mean = np.mean(delays)
        median = np.median(delays)
        p95 = np.percentile(delays, 95)
        print(f"{label:<30} {mean:<12.1f} {median:<12.1f} {p95:<12.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare packet delays between routing methods",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--batch-dir",
        required=True,
        help="Batch results directory containing run subdirectories (e.g., batch_results_20260202_132004)"
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
    print("DELAY COMPARISON TOOL")
    print("="*60)
    print(f"\nBatch directory: {batch_dir.resolve()}")
    print(f"Found {len(run_dirs)} run(s):")
    for run_dir in run_dirs:
        print(f"  - {run_dir.name}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")
    
    # Aggregate delay data from all runs
    print("\nAggregating data from all runs...")
    sensor_data = aggregate_delay_data(run_dirs, "sensor_packets")
    dm_data = aggregate_delay_data(run_dirs, "dm_packets")
    broadcast_data = aggregate_delay_data(run_dirs, "broadcast_packets")
    
    if not sensor_data and not dm_data and not broadcast_data:
        print("\nError: No delay CSV files found in run directories!")
        print("\nExpected file patterns in each run directory:")
        print("  - sensor_packets_<routing>.csv")
        print("  - dm_packets_<routing>.csv")
        print("  - broadcast_packets_<routing>.csv")
        sys.exit(1)
    
    # Compare each packet type
    # if sensor_data:
    #     print(f"\nFound sensor delay data for routing types:")
    #     for routing in sensor_data.keys():
    #         print(f"  - {routing}")
    #     compare_delays_overview(sensor_data, "sensor", args.output_dir, len(run_dirs))
    
    if dm_data:
        print(f"\nFound DM delay data for routing types:")
        for routing in dm_data.keys():
            print(f"  - {routing}")
        compare_delays_overview(dm_data, "dm", args.output_dir, len(run_dirs))
    
    if broadcast_data:
        print(f"\nFound broadcast delay data for routing types:")
        for routing in broadcast_data.keys():
            print(f"  - {routing}")
        compare_delays_overview(broadcast_data, "broadcast", args.output_dir, len(run_dirs))
    
    # Create overall summary
    create_summary_comparison(sensor_data, dm_data, broadcast_data, args.output_dir)
    
    print("\n" + "="*60)
    print("DELAY COMPARISON COMPLETE!")
    print("="*60)
    print(f"\nAll plots saved to: {Path(args.output_dir).resolve()}")
    print(f"Results aggregated across {len(run_dirs)} run(s)")


if __name__ == "__main__":
    main()
