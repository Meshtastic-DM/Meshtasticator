"""
Compare the number of alive nodes across different routing protocols.
A node is considered alive if its power > 0.
Calculates median across multiple runs for robustness.
"""

import pandas as pd
import numpy as np
import os
import sys

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

def count_alive_nodes(df):
    """
    Count the number of alive nodes (power > 0) at each time step.
    
    Args:
        df: DataFrame with time_ms column and node columns
        
    Returns:
        DataFrame with time_ms and alive_count columns
    """
    # Get all node columns (exclude time_ms)
    node_columns = [col for col in df.columns if col.startswith('node_')]
    
    # Count alive nodes (power > 0) for each row
    alive_counts = (df[node_columns] > 0).sum(axis=1)
    
    # Create result dataframe
    result = pd.DataFrame({
        'time_ms': df['time_ms'],
        'alive_count': alive_counts
    })
    
    return result

def plot_alive_nodes(protocols_data, output_file=None, show_quartiles=True):
    """
    Plot the number of alive nodes for all protocols on the same graph.
    
    Args:
        protocols_data: Dictionary with protocol names as keys and DataFrames as values
                       DataFrame should have 'time_ms', 'median', 'q25', 'q75' columns
        output_file: Optional path to save the plot
        show_quartiles: Whether to show 25th and 75th percentile bands
    """
    plt.figure(figsize=(14, 8))
    
    # Plot each protocol
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (protocol, df) in enumerate(protocols_data.items()):
        # Convert time from milliseconds to seconds or minutes for better readability
        time_minutes = df['time_ms'] / 60000  # Convert to minutes
        color = colors[i % len(colors)]
        
        # Plot median line
        plt.plot(time_minutes, df['median'], 
                label=f'{protocol} (median)', 
                linewidth=2.5, 
                marker='o', 
                markersize=5,
                color=color,
                alpha=0.9)
        
        # Add shaded area for quartiles if available and requested
        if show_quartiles and 'q25' in df.columns and 'q75' in df.columns:
            plt.fill_between(time_minutes, df['q25'], df['q75'], 
                           alpha=0.2, color=color)
    
    plt.xlabel('Time (minutes)', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Alive Nodes', fontsize=13, fontweight='bold')
    
    title = 'Network Node Survival Comparison Across Routing Protocols'
    if show_quartiles:
        title += '\n(Median with 25th-75th Percentile Range)'
    else:
        title += '\n(Median across runs)'
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc='best', fontsize=11, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add minor gridlines for better readability
    plt.grid(True, which='minor', alpha=0.2, linestyle=':')
    plt.minorticks_on()
    
    # Set y-axis to start from 0
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    
    plt.show()

def main(batch_folder=None):
    """
    Main function to compare alive nodes across protocols using median across multiple runs.
    
    Args:
        batch_folder: Path to the batch_results folder (e.g., batch_results_20260212_005354)
                     If None, will use the most recent batch_results folder
    """
    # Find the batch folder if not specified
    if batch_folder is None:
        # Look for batch_results folders
        workspace_path = os.path.dirname(os.path.abspath(__file__))
        batch_folders = [f for f in os.listdir(workspace_path) 
                        if f.startswith('batch_results_') and os.path.isdir(os.path.join(workspace_path, f))]
        
        if not batch_folders:
            print("Error: No batch_results folders found!")
            sys.exit(1)
        
        # Use the most recent one
        batch_folders.sort(reverse=True)
        batch_folder = os.path.join(workspace_path, batch_folders[0])
        print(f"Using batch folder: {batch_folder}")
    
    # Check if folder exists
    if not os.path.exists(batch_folder):
        print(f"Error: Folder {batch_folder} does not exist!")
        sys.exit(1)
    
    # Find all run folders
    run_folders = [f for f in os.listdir(batch_folder) 
                   if f.startswith('run_') and os.path.isdir(os.path.join(batch_folder, f))]
    
    if not run_folders:
        print(f"Error: No run folders found in {batch_folder}")
        sys.exit(1)
    
    run_folders.sort()
    print(f"Found {len(run_folders)} runs: {', '.join(run_folders)}")
    
    # Dictionary to store all runs data for each protocol
    # Structure: {protocol: {run: alive_df}}
    all_protocols_runs = {}
    
    # Load data from all runs
    for run_folder in run_folders:
        run_path = os.path.join(batch_folder, run_folder)
        
        # Find all protocol folders in this run
        protocol_folders = [f for f in os.listdir(run_path) 
                           if os.path.isdir(os.path.join(run_path, f)) and f != '__pycache__']
        
        for protocol in protocol_folders:
            protocol_path = os.path.join(run_path, protocol)
            
            # Find battery CSV file
            battery_files = [f for f in os.listdir(protocol_path) 
                            if f.startswith('battery_all_nodes_') and f.endswith('.csv')]
            
            if not battery_files:
                print(f"Warning: No battery CSV file found for {protocol} in {run_folder}")
                continue
            
            battery_file = os.path.join(protocol_path, battery_files[0])
            
            # Load CSV
            df = pd.read_csv(battery_file)
            
            # Count alive nodes
            alive_df = count_alive_nodes(df)
            
            # Store the data
            if protocol not in all_protocols_runs:
                all_protocols_runs[protocol] = {}
            all_protocols_runs[protocol][run_folder] = alive_df
            
            print(f"Loaded {run_folder}/{protocol}: {alive_df['alive_count'].iloc[0]} → {alive_df['alive_count'].iloc[-1]} alive nodes")
    
    if not all_protocols_runs:
        print("Error: No valid protocol data found!")
        sys.exit(1)
    
    print(f"\nFound protocols: {', '.join(all_protocols_runs.keys())}")
    
    # Calculate median and quartiles across runs for each protocol
    protocols_median_data = {}
    
    for protocol, runs_data in all_protocols_runs.items():
        print(f"\nProcessing {protocol} ({len(runs_data)} runs)...")
        
        # Combine all runs data
        all_runs_list = []
        for run_name, df in runs_data.items():
            df_copy = df.copy()
            df_copy['run'] = run_name
            all_runs_list.append(df_copy)
        
        combined_df = pd.concat(all_runs_list, ignore_index=True)
        
        # Group by time_ms and calculate statistics
        stats_df = combined_df.groupby('time_ms')['alive_count'].agg([
            ('median', 'median'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('q25', lambda x: np.percentile(x, 25)),
            ('q75', lambda x: np.percentile(x, 75)),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).reset_index()
        
        protocols_median_data[protocol] = stats_df
        
        # Print summary
        initial_median = stats_df['median'].iloc[0]
        final_median = stats_df['median'].iloc[-1]
        print(f"  Median: {initial_median:.1f} → {final_median:.1f} alive nodes")
        print(f"  Data points per time step: {stats_df['count'].iloc[0]}")
    
    # Create output filename
    output_file = os.path.join(batch_folder, 'alive_nodes_comparison_median.png')
    
    # Plot the comparison
    print("\nGenerating plot...")
    plot_alive_nodes(protocols_median_data, output_file, show_quartiles=True)
    
    # Save CSV summary
    summary_file = os.path.join(batch_folder, 'alive_nodes_summary_median.csv')
    
    # Merge all protocol data into one DataFrame
    merged_df = None
    for protocol, df in protocols_median_data.items():
        # Select relevant columns and rename
        df_subset = df[['time_ms', 'median', 'mean', 'q25', 'q75', 'min', 'max']].copy()
        df_subset.columns = ['time_ms'] + [f'{protocol}_{col}' for col in df_subset.columns[1:]]
        
        if merged_df is None:
            merged_df = df_subset
        else:
            merged_df = merged_df.merge(df_subset, on='time_ms', how='outer')
    
    merged_df = merged_df.sort_values('time_ms')
    merged_df.to_csv(summary_file, index=False)
    print(f"Summary CSV saved to: {summary_file}")
    print(f"\nAnalysis complete! Used {len(run_folders)} runs for median calculation.")

if __name__ == '__main__':
    # Check if batch folder is provided as argument
    if len(sys.argv) > 1:
        batch_folder = sys.argv[1]
        main(batch_folder)
    else:
        main()
