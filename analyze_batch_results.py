#!/usr/bin/env python3
"""
Batch Results Analyzer

Analyzes batch simulation results and computes average metrics across all runs.
Generates summary statistics for sensor delay, sensor reliability, DM delay,
DM reliability, broadcast delay, broadcast reliability, and energy consumption.

Usage:
    python analyze_batch_results.py --batch batch_results_20251211_120756
    python analyze_batch_results.py --batch batch_results_20251211_120756 --routing AODV SDN_AODV
    python analyze_batch_results.py --all  # Analyze all batch_results_* directories
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from datetime import datetime


def read_csv_data(csv_path):
    """Read CSV file and return data as list of dictionaries."""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        print(f"    Warning: Could not read {csv_path.name}: {e}")
        return []


def calculate_sensor_metrics(run_dir, routing_type):
    """Calculate average sensor delay and reliability."""
    metrics = {
        'delay': None,
        'reliability': None
    }
    
    # Sensor delay from sensor_packets CSV
    delay_file = run_dir / routing_type / f"sensor_packets_ROUTER_TYPE.{routing_type}.csv"
    if delay_file.exists():
        data = read_csv_data(delay_file)
        if data:
            delays = [float(row['delay_value']) for row in data if 'delay_value' in row]
            if delays:
                metrics['delay'] = np.mean(delays)
    
    # Sensor reliability from sensor_reliability CSV
    reliability_file = run_dir / routing_type / f"sensor_reliability_to_dest0_ROUTER_TYPE.{routing_type}.csv"
    if reliability_file.exists():
        data = read_csv_data(reliability_file)
        if data:
            reliabilities = []
            for row in data:
                if 'reliability_to_dest_0' in row and row['reliability_to_dest_0']:
                    try:
                        reliabilities.append(float(row['reliability_to_dest_0']))
                    except ValueError:
                        pass
            if reliabilities:
                metrics['reliability'] = np.mean(reliabilities)
    
    return metrics


def calculate_dm_metrics(run_dir, routing_type):
    """Calculate average DM delay and reliability."""
    metrics = {
        'delay': None,
        'reliability': None
    }
    
    # DM delay from dm_packets CSV
    delay_file = run_dir / routing_type / f"dm_packets_ROUTER_TYPE.{routing_type}.csv"
    if delay_file.exists():
        data = read_csv_data(delay_file)
        if data:
            delays = [float(row['delay_value']) for row in data if 'delay_value' in row]
            if delays:
                metrics['delay'] = np.mean(delays)
    
    # DM reliability from dm_reliability_matrix CSV
    reliability_file = run_dir / routing_type / f"dm_reliability_matrix_ROUTER_TYPE.{routing_type}.csv"
    if reliability_file.exists():
        data = read_csv_data(reliability_file)
        if data:
            reliabilities = []
            for row in data:
                # Get all values except the first column (source node ID)
                for key, value in row.items():
                    if key != list(row.keys())[0] and value:  # Skip first column (node IDs)
                        try:
                            reliabilities.append(float(value))
                        except ValueError:
                            pass
            if reliabilities:
                metrics['reliability'] = np.mean(reliabilities)
    
    return metrics


def calculate_broadcast_metrics(run_dir, routing_type):
    """Calculate average broadcast delay and reliability."""
    metrics = {
        'delay': None,
        'reliability': None
    }
    
    # Broadcast delay from broadcast_packets CSV
    delay_file = run_dir / routing_type / f"broadcast_packets_ROUTER_TYPE.{routing_type}.csv"
    if delay_file.exists():
        data = read_csv_data(delay_file)
        if data:
            delays = [float(row['delay_value']) for row in data if 'delay_value' in row]
            if delays:
                metrics['delay'] = np.mean(delays)
    
    # Broadcast reliability from broadcast_reliability CSV
    reliability_file = run_dir / routing_type / f"broadcast_reliability_ROUTER_TYPE.{routing_type}.csv"
    if reliability_file.exists():
        data = read_csv_data(reliability_file)
        if data:
            reliabilities = []
            for row in data:
                # Try both possible column names
                value = row.get('reliability') or row.get('reliability_from_src_0')
                if value:
                    try:
                        reliabilities.append(float(value))
                    except ValueError:
                        pass
            if reliabilities:
                metrics['reliability'] = np.mean(reliabilities)
    
    return metrics


def calculate_energy_consumption(run_dir, routing_type):
    """Calculate average energy consumption."""
    energy_file = run_dir / routing_type / f"energy_consumption_ROUTER_TYPE.{routing_type}.csv"
    
    if energy_file.exists():
        data = read_csv_data(energy_file)
        if data:
            # Get energy consumption (check for different possible column names)
            energies = []
            for row in data:
                # Try different column names
                value = row.get('energy_consumed_J') or row.get('total_energy') or row.get('energy_consumption')
                if value:
                    try:
                        energies.append(float(value))
                    except ValueError:
                        pass
            if energies:
                return np.mean(energies)
    
    return None


def analyze_batch(batch_dir, routing_types=None):
    """
    Analyze a single batch results directory.
    
    Args:
        batch_dir: Path to batch results directory
        routing_types: List of routing types to analyze (None = all)
    
    Returns:
        Dictionary with analysis results
    """
    batch_dir = Path(batch_dir)
    
    if not batch_dir.exists():
        print(f"Error: Batch directory not found: {batch_dir}")
        return None
    
    print(f"\nAnalyzing: {batch_dir.name}")
    print("-" * 80)
    
    # Find all run directories
    run_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
    
    if not run_dirs:
        print(f"  No run directories found in {batch_dir}")
        return None
    
    print(f"  Found {len(run_dirs)} runs")
    
    # Determine routing types to analyze
    if routing_types is None:
        # Auto-detect from first run
        first_run = run_dirs[0]
        routing_types = [d.name for d in first_run.iterdir() if d.is_dir()]
    
    print(f"  Routing types: {', '.join(routing_types)}")
    
    # Collect metrics for each routing type
    results = {}
    
    for routing_type in routing_types:
        print(f"\n  Processing {routing_type}...")
        
        routing_metrics = {
            'sensor_delay': [],
            'sensor_reliability': [],
            'dm_delay': [],
            'dm_reliability': [],
            'broadcast_delay': [],
            'broadcast_reliability': [],
            'energy_consumption': []
        }
        
        runs_processed = 0
        
        for run_dir in run_dirs:
            routing_dir = run_dir / routing_type
            
            if not routing_dir.exists():
                continue
            
            runs_processed += 1
            
            # Calculate sensor metrics
            sensor = calculate_sensor_metrics(run_dir, routing_type)
            if sensor['delay'] is not None:
                routing_metrics['sensor_delay'].append(sensor['delay'])
            if sensor['reliability'] is not None:
                routing_metrics['sensor_reliability'].append(sensor['reliability'])
            
            # Calculate DM metrics
            dm = calculate_dm_metrics(run_dir, routing_type)
            if dm['delay'] is not None:
                routing_metrics['dm_delay'].append(dm['delay'])
            if dm['reliability'] is not None:
                routing_metrics['dm_reliability'].append(dm['reliability'])
            
            # Calculate broadcast metrics
            broadcast = calculate_broadcast_metrics(run_dir, routing_type)
            if broadcast['delay'] is not None:
                routing_metrics['broadcast_delay'].append(broadcast['delay'])
            if broadcast['reliability'] is not None:
                routing_metrics['broadcast_reliability'].append(broadcast['reliability'])
            
            # Calculate energy consumption
            energy = calculate_energy_consumption(run_dir, routing_type)
            if energy is not None:
                routing_metrics['energy_consumption'].append(energy)
        
        print(f"    Processed {runs_processed} runs")
        
        # Calculate averages and std deviations
        results[routing_type] = {
            'runs_processed': runs_processed,
            'metrics': {}
        }
        
        for metric_name, values in routing_metrics.items():
            if values:
                results[routing_type]['metrics'][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
            else:
                results[routing_type]['metrics'][metric_name] = None
    
    return results


def print_results(batch_name, results):
    """Print analysis results in a readable format."""
    print("\n" + "=" * 80)
    print(f"ANALYSIS RESULTS: {batch_name}")
    print("=" * 80)
    
    for routing_type, data in results.items():
        print(f"\n{routing_type}")
        print("-" * 40)
        print(f"Runs processed: {data['runs_processed']}")
        print()
        
        metrics = data['metrics']
        
        # Print each metric
        metric_names = [
            ('sensor_delay', 'Sensor Delay (ms)'),
            ('sensor_reliability', 'Sensor Reliability'),
            ('dm_delay', 'DM Delay (ms)'),
            ('dm_reliability', 'DM Reliability'),
            ('broadcast_delay', 'Broadcast Delay (ms)'),
            ('broadcast_reliability', 'Broadcast Reliability'),
            ('energy_consumption', 'Energy Consumption (mJ)')
        ]
        
        for metric_key, metric_label in metric_names:
            if metrics[metric_key] is not None:
                m = metrics[metric_key]
                print(f"  {metric_label}:")
                print(f"    Mean: {m['mean']:.4f}")
                print(f"    Std:  {m['std']:.4f}")
                print(f"    Min:  {m['min']:.4f}")
                print(f"    Max:  {m['max']:.4f}")
                print(f"    Count: {m['count']}")
                print()
            else:
                print(f"  {metric_label}: No data")
                print()


def save_results_to_csv(batch_name, results, output_file):
    """Save analysis results to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Batch', 'Routing Type', 'Runs Processed',
            'Sensor Delay Mean', 'Sensor Delay Std',
            'Sensor Reliability Mean', 'Sensor Reliability Std',
            'DM Delay Mean', 'DM Delay Std',
            'DM Reliability Mean', 'DM Reliability Std',
            'Broadcast Delay Mean', 'Broadcast Delay Std',
            'Broadcast Reliability Mean', 'Broadcast Reliability Std',
            'Energy Consumption Mean', 'Energy Consumption Std'
        ])
        
        # Data rows
        for routing_type, data in results.items():
            metrics = data['metrics']
            
            row = [
                batch_name,
                routing_type,
                data['runs_processed']
            ]
            
            # Add metrics
            for metric_key in ['sensor_delay', 'sensor_reliability', 'dm_delay', 'dm_reliability',
                              'broadcast_delay', 'broadcast_reliability', 'energy_consumption']:
                if metrics[metric_key] is not None:
                    row.extend([
                        f"{metrics[metric_key]['mean']:.4f}",
                        f"{metrics[metric_key]['std']:.4f}"
                    ])
                else:
                    row.extend(['N/A', 'N/A'])
            
            writer.writerow(row)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze batch simulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--batch', type=str, help='Specific batch directory to analyze')
    group.add_argument('--all', action='store_true', help='Analyze all batch_results_* directories')
    
    parser.add_argument(
        '--routing',
        nargs='+',
        help='Specific routing types to analyze (default: all found)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for results (default: batch_analysis_TIMESTAMP.csv)'
    )
    
    args = parser.parse_args()
    
    # Determine batch directories to analyze
    if args.all:
        batch_dirs = sorted([d for d in Path('.').iterdir() 
                           if d.is_dir() and d.name.startswith('batch_results_')])
        if not batch_dirs:
            print("No batch_results_* directories found")
            return
        print(f"Found {len(batch_dirs)} batch directories")
    else:
        batch_dirs = [Path(args.batch)]
    
    # Analyze each batch
    all_results = {}
    
    for batch_dir in batch_dirs:
        results = analyze_batch(batch_dir, args.routing)
        if results:
            all_results[batch_dir.name] = results
            print_results(batch_dir.name, results)
    
    # Save to CSV if requested or analyzing multiple batches
    if args.output or args.all:
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"batch_analysis_{timestamp}.csv"
        
        # Save all results
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Batch', 'Routing Type', 'Runs Processed',
                'Sensor Delay Mean', 'Sensor Delay Std',
                'Sensor Reliability Mean', 'Sensor Reliability Std',
                'DM Delay Mean', 'DM Delay Std',
                'DM Reliability Mean', 'DM Reliability Std',
                'Broadcast Delay Mean', 'Broadcast Delay Std',
                'Broadcast Reliability Mean', 'Broadcast Reliability Std',
                'Energy Consumption Mean', 'Energy Consumption Std'
            ])
            
            # Write data for all batches
            for batch_name, results in all_results.items():
                for routing_type, data in results.items():
                    metrics = data['metrics']
                    
                    row = [
                        batch_name,
                        routing_type,
                        data['runs_processed']
                    ]
                    
                    for metric_key in ['sensor_delay', 'sensor_reliability', 'dm_delay', 'dm_reliability',
                                      'broadcast_delay', 'broadcast_reliability', 'energy_consumption']:
                        if metrics[metric_key] is not None:
                            row.extend([
                                f"{metrics[metric_key]['mean']:.4f}",
                                f"{metrics[metric_key]['std']:.4f}"
                            ])
                        else:
                            row.extend(['N/A', 'N/A'])
                    
                    writer.writerow(row)
        
        print(f"\n{'=' * 80}")
        print(f"All results saved to: {output_file}")
        print("=" * 80)


if __name__ == "__main__":
    main()
