#!/usr/bin/env python3
"""
Broadcast Message Interval Comparison Plotter

Plots broadcast delay and reliability metrics for different message generation intervals
comparing multiple routing approaches (e.g., AODV vs MANAGED_FLOOD).

Usage:
    python plot_broadcast_interval_comparison.py --input boradcastmessages_generation.csv
    python plot_broadcast_interval_comparison.py --input boradcastmessages_generation.csv --output comparison.png
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def read_broadcast_data(csv_file):
    """Read broadcast message interval data from CSV."""
    data = {}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            interval = float(row['Boradcast_message_expected_interval'])
            routing = row['Routing']
            delay = float(row['Broadcast_Delay'])
            reliability = float(row['Broadcast_Reliability'])
            
            if routing not in data:
                data[routing] = {
                    'intervals': [],
                    'delays': [],
                    'reliabilities': []
                }
            
            data[routing]['intervals'].append(interval)
            data[routing]['delays'].append(delay)
            data[routing]['reliabilities'].append(reliability)
    
    return data


def plot_comparison(data, output_file=None):
    """Create comparison plots for different routing approaches."""
    
    # Create figure with 2 subplots (delay and reliability)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors and markers for different routing types
    colors = {'AODV': '#1f77b4', 'MANAGED_FLOOD': '#ff7f0e', 'SDN_AODV': '#2ca02c'}
    markers = {'AODV': 'o', 'MANAGED_FLOOD': 's', 'SDN_AODV': '^'}
    
    routing_types = list(data.keys())
    
    # Plot 1: Broadcast Delay vs Message Interval
    for routing in routing_types:
        intervals = data[routing]['intervals']
        delays = data[routing]['delays']
        
        ax1.plot(intervals, delays, 
                marker=markers.get(routing, 'o'),
                markersize=8,
                linewidth=2.5,
                label=routing,
                color=colors.get(routing, None))
    
    ax1.set_xlabel('Broadcast Message Generation Interval (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Broadcast Delay (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Broadcast Delay vs Message Generation Interval', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=10)
    
    # Plot 2: Broadcast Reliability vs Message Interval
    for routing in routing_types:
        intervals = data[routing]['intervals']
        reliabilities = data[routing]['reliabilities']
        
        ax2.plot(intervals, reliabilities,
                marker=markers.get(routing, 'o'),
                markersize=8,
                linewidth=2.5,
                label=routing,
                color=colors.get(routing, None))
    
    ax2.set_xlabel('Broadcast Message Generation Interval (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Broadcast Reliability (0-1)', fontsize=12, fontweight='bold')
    ax2.set_title('Broadcast Reliability vs Message Generation Interval', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_file}")
    else:
        plt.savefig('broadcast_interval_comparison.png', dpi=300, bbox_inches='tight')
        print("Comparison plot saved to: broadcast_interval_comparison.png")
    
    # Show plot
    plt.show()


def plot_combined_metric(data, output_file=None):
    """Create a combined metric plot (delay normalized + reliability)."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'AODV': '#1f77b4', 'MANAGED_FLOOD': '#ff7f0e', 'SDN_AODV': '#2ca02c'}
    markers = {'AODV': 'o', 'MANAGED_FLOOD': 's', 'SDN_AODV': '^'}
    
    routing_types = list(data.keys())
    
    # Calculate normalized combined score: reliability / (normalized_delay)
    # Higher is better
    for routing in routing_types:
        intervals = data[routing]['intervals']
        delays = data[routing]['delays']
        reliabilities = data[routing]['reliabilities']
        
        # Normalize delays to 0-1 range (invert: lower delay = higher score)
        max_delay = max(delays)
        normalized_delays = [1 - (d / max_delay) for d in delays]
        
        # Combined score: average of reliability and normalized delay
        combined_scores = [(r + nd) / 2 for r, nd in zip(reliabilities, normalized_delays)]
        
        ax.plot(intervals, combined_scores,
                marker=markers.get(routing, 'o'),
                markersize=8,
                linewidth=2.5,
                label=routing,
                color=colors.get(routing, None))
    
    ax.set_xlabel('Broadcast Message Generation Interval (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Combined Performance Score (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Combined Performance (Reliability + Normalized Delay)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Combined metric plot saved to: {output_file}")
    else:
        plt.savefig('broadcast_combined_score.png', dpi=300, bbox_inches='tight')
        print("Combined metric plot saved to: broadcast_combined_score.png")
    
    plt.show()


def plot_bar_comparison(data, output_file=None):
    """Create bar chart comparison for each interval."""
    
    routing_types = list(data.keys())
    intervals = data[routing_types[0]]['intervals']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(intervals))
    width = 0.35 if len(routing_types) == 2 else 0.25
    
    colors = {'AODV': '#1f77b4', 'MANAGED_FLOOD': '#ff7f0e', 'SDN_AODV': '#2ca02c'}
    
    # Bar chart 1: Delays
    for i, routing in enumerate(routing_types):
        delays = data[routing]['delays']
        offset = width * (i - len(routing_types)/2 + 0.5)
        ax1.bar(x + offset, delays, width, label=routing, color=colors.get(routing, None))
    
    ax1.set_xlabel('Broadcast Message Generation Interval (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Broadcast Delay (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Broadcast Delay Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{int(iv)}s" for iv in intervals])
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(labelsize=10)
    
    # Bar chart 2: Reliability
    for i, routing in enumerate(routing_types):
        reliabilities = data[routing]['reliabilities']
        offset = width * (i - len(routing_types)/2 + 0.5)
        ax2.bar(x + offset, reliabilities, width, label=routing, color=colors.get(routing, None))
    
    ax2.set_xlabel('Broadcast Message Generation Interval (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Broadcast Reliability (0-1)', fontsize=12, fontweight='bold')
    ax2.set_title('Broadcast Reliability Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(iv)}s" for iv in intervals])
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    # Save plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Bar comparison plot saved to: {output_file}")
    else:
        plt.savefig('broadcast_bar_comparison.png', dpi=300, bbox_inches='tight')
        print("Bar comparison plot saved to: broadcast_bar_comparison.png")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot broadcast message interval comparison between routing approaches',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file with broadcast data'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file prefix for plots (default: broadcast_interval_comparison.png)'
    )
    
    parser.add_argument(
        '--plot-type',
        type=str,
        choices=['line', 'bar', 'combined', 'all'],
        default='line',
        help='Type of plot to generate (default: line)'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Read data
    print(f"Reading data from: {args.input}")
    data = read_broadcast_data(args.input)
    
    print(f"Found {len(data)} routing types:")
    for routing in data.keys():
        print(f"  - {routing}: {len(data[routing]['intervals'])} data points")
    
    # Generate plots
    if args.plot_type == 'line' or args.plot_type == 'all':
        output = args.output if args.output else None
        plot_comparison(data, output)
    
    if args.plot_type == 'bar' or args.plot_type == 'all':
        output = f"{args.output.replace('.png', '')}_bar.png" if args.output else None
        plot_bar_comparison(data, output)
    
    if args.plot_type == 'combined' or args.plot_type == 'all':
        output = f"{args.output.replace('.png', '')}_combined.png" if args.output else None
        plot_combined_metric(data, output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
