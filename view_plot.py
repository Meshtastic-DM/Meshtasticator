#!/usr/bin/env python3
"""
Interactive Plot Viewer for Pickled Matplotlib Figures

Usage:
    python view_plot.py <pickle_file_path>
    
Example:
    python view_plot.py output/sensor_reliability_AODV.pkl
    python view_plot.py batch_results_20251208_222855/run_001/SDN_AODV/sensor_reliability_SDN_AODV.pkl
"""

import argparse
import pickle
import sys
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


def main():
    parser = argparse.ArgumentParser(
        description='View pickled matplotlib figures interactively',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python view_plot.py output/sensor_reliability_AODV.pkl
  python view_plot.py batch_results_20251208_222855/run_001/SDN_AODV/dm_reliability_matrix_SDN_AODV.pkl
        '''
    )
    parser.add_argument(
        'pickle_file',
        type=str,
        help='Path to the pickled figure file (.pkl)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pickle_file):
        print(f"Error: File not found: {args.pickle_file}")
        sys.exit(1)
    
    # Check file extension
    if not args.pickle_file.endswith('.pkl'):
        print(f"Warning: File does not have .pkl extension: {args.pickle_file}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    try:
        # Load the pickled figure
        print(f"Loading figure from: {args.pickle_file}")
        with open(args.pickle_file, 'rb') as f:
            fig = pickle.load(f)
        
        print("Figure loaded successfully!")
        print("You can now zoom, pan, and interact with the plot.")
        print("Close the window to exit.")
        
        # Display the figure interactively
        plt.show()
        
    except Exception as e:
        print(f"Error loading or displaying figure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
