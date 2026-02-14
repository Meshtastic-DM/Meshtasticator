"""
Compare packet delays between different routing methods.

This script loads delay CSV files from multiple routing types and creates
comprehensive delay comparison visualizations using box plots, violin plots, and bar charts.

Usage:
    python compare_delays.py
    
    Or specify custom directory:
    python compare_delays.py --input-dir output/
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


def compare_delays_overview(files_dict, packet_type, output_dir="."):
    """
    Create overview comparison of delays across routing methods.
    
    Args:
        files_dict: dict of {routing_name: filepath}
        packet_type: 'sensor', 'dm', or 'broadcast'
        output_dir: directory to save output plots
    """
    print(f"\n{'='*60}")
    print(f"{packet_type.upper()} PACKET DELAY COMPARISON")
    print(f"{'='*60}")
    
    data = {}
    for routing_name, filepath in files_dict.items():
        df = load_delay_csv(filepath)
        if df is not None and len(df) > 0:
            data[routing_name] = df
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


def create_summary_comparison(sensor_files, dm_files, broadcast_files, output_dir="."):
    """Create a single summary plot comparing all packet types and routing methods."""
    print(f"\n{'='*60}")
    print("OVERALL DELAY SUMMARY")
    print(f"{'='*60}")
    
    all_data = {}
    
    # Load all data
    for packet_type, files_dict in [('Sensor', sensor_files), ('DM', dm_files), ('Broadcast', broadcast_files)]:
        if not files_dict:
            continue
        for routing_name, filepath in files_dict.items():
            df = load_delay_csv(filepath)
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
        "--input-dir",
        default="output",
        help="Directory containing delay CSV files (default: output)"
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
    
    # Auto-detect delay files
    sensor_files = {}
    dm_files = {}
    broadcast_files = {}
    
    for csv_file in input_dir.glob("*packets*.csv"):
        filename = csv_file.name
        
        # Skip reliability files
        if "reliability" in filename:
            continue
        
        # Extract routing type from filename
        if "ROUTER_TYPE." in filename:
            routing_type = filename.split("ROUTER_TYPE.")[1].replace(".csv", "")
        else:
            continue
        
        if "sensor_packets" in filename:
            sensor_files[routing_type] = str(csv_file)
        elif "dm_packets" in filename:
            dm_files[routing_type] = str(csv_file)
        elif "broadcast_packets" in filename:
            broadcast_files[routing_type] = str(csv_file)
    
    if not sensor_files and not dm_files and not broadcast_files:
        print(f"No delay CSV files found in '{input_dir}'!")
        print("\nExpected file patterns:")
        print("  - sensor_packets_ROUTER_TYPE.<routing>.csv")
        print("  - dm_packets_ROUTER_TYPE.<routing>.csv")
        print("  - broadcast_packets_ROUTER_TYPE.<routing>.csv")
        sys.exit(1)
    
    print("="*60)
    print("DELAY COMPARISON TOOL")
    print("="*60)
    print(f"\nInput directory: {input_dir.resolve()}")
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")
    
    # Compare each packet type
    if sensor_files:
        print(f"\nFound sensor packet files:")
        for routing, path in sensor_files.items():
            print(f"  - {routing}: {Path(path).name}")
        compare_delays_overview(sensor_files, "sensor", args.output_dir)
    
    if dm_files:
        print(f"\nFound DM packet files:")
        for routing, path in dm_files.items():
            print(f"  - {routing}: {Path(path).name}")
        compare_delays_overview(dm_files, "dm", args.output_dir)
    
    if broadcast_files:
        print(f"\nFound broadcast packet files:")
        for routing, path in broadcast_files.items():
            print(f"  - {routing}: {Path(path).name}")
        compare_delays_overview(broadcast_files, "broadcast", args.output_dir)
    
    # Create overall summary
    create_summary_comparison(sensor_files, dm_files, broadcast_files, args.output_dir)
    
    print("\n" + "="*60)
    print("DELAY COMPARISON COMPLETE!")
    print("="*60)
    print(f"\nAll plots saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
