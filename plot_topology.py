"""
Visualize network topology from config.yaml file.

This script reads the node configuration file and creates a scatter plot
showing the positions of all nodes with different colors for each node type.

Usage:
    python plot_topology.py
    
    Or specify custom config file:
    python plot_topology.py --config path/to/config.yaml
"""

import yaml
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


def load_config(config_path):
    """Load and parse the YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_nodes(config):
    """Extract node information from config dictionary."""
    nodes = []
    
    for node_id, node_data in config.items():
        if isinstance(node_data, dict) and 'x' in node_data:
            nodes.append({
                'id': node_id,
                'x': node_data.get('x', 0),
                'y': node_data.get('y', 0),
                'z': node_data.get('z', 1),
                'simRole': node_data.get('simRole', 'Unknown'),
                'isRouter': node_data.get('isRouter', False),
                'isRepeater': node_data.get('isRepeater', False)
            })
    
    return nodes


def plot_topology(nodes, output_file="network_topology.png", show_labels=True, show_grid=True):
    """
    Create a scatter plot of the network topology.
    
    Args:
        nodes: List of node dictionaries
        output_file: Path to save the plot
        show_labels: Whether to show node IDs as labels
        show_grid: Whether to show grid lines
    """
    # Define colors and markers for each node type
    role_styles = {
        'Control_Center': {'color': '#FF0000', 'marker': '*', 'size': 600, 'label': 'Control Center'},
        'DM': {'color': '#2E86DE', 'marker': 'o', 'size': 300, 'label': 'DM Node'},
        'Sensor': {'color': '#10AC84', 'marker': '^', 'size': 300, 'label': 'Sensor Node'},
        'Router': {'color': '#FF6348', 'marker': 's', 'size': 400, 'label': 'Router'},
        'Unknown': {'color': '#95A5A6', 'marker': 'x', 'size': 200, 'label': 'Unknown'}
    }
    
    # Group nodes by role
    nodes_by_role = {}
    for node in nodes:
        role = node['simRole']
        if role not in nodes_by_role:
            nodes_by_role[role] = []
        nodes_by_role[role].append(node)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot each group with its style
    for role, role_nodes in nodes_by_role.items():
        style = role_styles.get(role, role_styles['Unknown'])
        
        x_coords = [node['x'] for node in role_nodes]
        y_coords = [node['y'] for node in role_nodes]
        
        # Plot nodes
        ax.scatter(
            x_coords, 
            y_coords,
            c=style['color'],
            marker=style['marker'],
            s=style['size'],
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5,
            label=style['label'],
            zorder=3
        )
        
        # Add node ID labels
        if show_labels:
            for node in role_nodes:
                ax.annotate(
                    f"{node['id']}",
                    (node['x'], node['y']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'),
                    zorder=4
                )
    
    # Styling
    ax.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontsize=12, fontweight='bold')
    ax.set_title('Network Topology - Node Positions', fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
    
    # Grid
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Equal aspect ratio for accurate distances
    ax.set_aspect('equal', adjustable='box')
    
    # Add axis limits with padding
    all_x = [node['x'] for node in nodes]
    all_y = [node['y'] for node in nodes]
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    padding = max(x_range, y_range) * 0.1
    
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ Topology plot saved to: {output_file}")
    plt.close()


def plot_topology_with_connections(nodes, output_file="network_topology_with_range.png", 
                                   max_range=1000, show_labels=True):
    """
    Create a topology plot with potential connection lines based on distance.
    
    Args:
        nodes: List of node dictionaries
        output_file: Path to save the plot
        max_range: Maximum communication range in meters
        show_labels: Whether to show node IDs
    """
    # Define colors for each node type
    role_styles = {
        'Control_Center': {'color': '#FF0000', 'marker': '*', 'size': 600, 'label': 'Control Center'},
        'DM': {'color': '#2E86DE', 'marker': 'o', 'size': 300, 'label': 'DM Node'},
        'Sensor': {'color': '#10AC84', 'marker': '^', 'size': 300, 'label': 'Sensor Node'},
        'Router': {'color': '#FF6348', 'marker': 's', 'size': 400, 'label': 'Router'},
        'Unknown': {'color': '#95A5A6', 'marker': 'x', 'size': 200, 'label': 'Unknown'}
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw connection lines first (so they appear behind nodes)
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            distance = np.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)
            if distance <= max_range:
                ax.plot(
                    [node1['x'], node2['x']], 
                    [node1['y'], node2['y']], 
                    'gray', 
                    alpha=0.15, 
                    linewidth=0.5,
                    zorder=1
                )
    
    # Group nodes by role
    nodes_by_role = {}
    for node in nodes:
        role = node['simRole']
        if role not in nodes_by_role:
            nodes_by_role[role] = []
        nodes_by_role[role].append(node)
    
    # Plot each group
    for role, role_nodes in nodes_by_role.items():
        style = role_styles.get(role, role_styles['Unknown'])
        
        x_coords = [node['x'] for node in role_nodes]
        y_coords = [node['y'] for node in role_nodes]
        
        ax.scatter(
            x_coords, 
            y_coords,
            c=style['color'],
            marker=style['marker'],
            s=style['size'],
            alpha=0.8,
            edgecolors='black',
            linewidth=1.5,
            label=style['label'],
            zorder=3
        )
        
        if show_labels:
            for node in role_nodes:
                ax.annotate(
                    f"{node['id']}",
                    (node['x'], node['y']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
                    zorder=4
                )
    
    # Styling
    ax.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'Network Topology with Potential Links (Range: {max_range}m)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Set limits
    all_x = [node['x'] for node in nodes]
    all_y = [node['y'] for node in nodes]
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    padding = max(x_range, y_range) * 0.1
    
    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ Topology with connections saved to: {output_file}")
    plt.close()


def plot_topology_3d(nodes, output_file="network_topology_3d.png", show_labels=True):
    """
    Create a 3D scatter plot of the network topology.
    
    Args:
        nodes: List of node dictionaries
        output_file: Path to save the plot
        show_labels: Whether to show node IDs
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    role_styles = {
        'Control_Center': {'color': '#FF0000', 'marker': '*', 'size': 600, 'label': 'Control Center'},
        'DM': {'color': '#2E86DE', 'marker': 'o', 'size': 300, 'label': 'DM Node'},
        'Sensor': {'color': '#10AC84', 'marker': '^', 'size': 300, 'label': 'Sensor Node'},
        'Router': {'color': '#FF6348', 'marker': 's', 'size': 400, 'label': 'Router'},
        'Unknown': {'color': '#95A5A6', 'marker': 'x', 'size': 200, 'label': 'Unknown'}
    }
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Group nodes by role
    nodes_by_role = {}
    for node in nodes:
        role = node['simRole']
        if role not in nodes_by_role:
            nodes_by_role[role] = []
        nodes_by_role[role].append(node)
    
    # Plot each group
    for role, role_nodes in nodes_by_role.items():
        style = role_styles.get(role, role_styles['Unknown'])
        
        x_coords = [node['x'] for node in role_nodes]
        y_coords = [node['y'] for node in role_nodes]
        z_coords = [node['z'] for node in role_nodes]
        
        ax.scatter(
            x_coords, 
            y_coords,
            z_coords,
            c=style['color'],
            marker=style['marker'],
            s=style['size'],
            alpha=0.8,
            edgecolors='black',
            linewidth=1.5,
            label=style['label']
        )
        
        if show_labels:
            for node in role_nodes:
                ax.text(
                    node['x'], 
                    node['y'], 
                    node['z'],
                    f"  {node['id']}",
                    fontsize=9,
                    fontweight='bold'
                )
    
    ax.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Coordinate (m)', fontsize=12, fontweight='bold')
    ax.set_title('Network Topology - 3D View', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"✓ 3D topology plot saved to: {output_file}")
    plt.close()


def print_statistics(nodes):
    """Print summary statistics about the network."""
    print("\n" + "="*60)
    print("NETWORK TOPOLOGY STATISTICS")
    print("="*60)
    
    # Count by role
    role_counts = {}
    for node in nodes:
        role = node['simRole']
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print(f"\nTotal Nodes: {len(nodes)}")
    print("\nNodes by Type:")
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count}")
    
    # Calculate network dimensions
    x_coords = [node['x'] for node in nodes]
    y_coords = [node['y'] for node in nodes]
    z_coords = [node['z'] for node in nodes]
    
    print(f"\nNetwork Dimensions:")
    print(f"  X range: {min(x_coords):.1f}m to {max(x_coords):.1f}m (span: {max(x_coords)-min(x_coords):.1f}m)")
    print(f"  Y range: {min(y_coords):.1f}m to {max(y_coords):.1f}m (span: {max(y_coords)-min(y_coords):.1f}m)")
    print(f"  Z range: {min(z_coords):.1f}m to {max(z_coords):.1f}m (span: {max(z_coords)-min(z_coords):.1f}m)")
    
    # Calculate average distances
    distances = []
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            dist = np.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)
            distances.append(dist)
    
    if distances:
        print(f"\nPairwise Distances:")
        print(f"  Average: {np.mean(distances):.1f}m")
        print(f"  Minimum: {min(distances):.1f}m")
        print(f"  Maximum: {max(distances):.1f}m")
        print(f"  Median: {np.median(distances):.1f}m")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize network topology from config.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        default="out/config.yaml",
        help="Path to config.yaml file (default: out/config.yaml)"
    )
    parser.add_argument(
        "--output-dir",
        default="output/plots",
        help="Directory to save plots (default: output/plots)"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't show node ID labels"
    )
    parser.add_argument(
        "--range",
        type=float,
        default=1000,
        help="Communication range in meters for connection plot (default: 1000)"
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        dest="plot_3d",
        help="Also create a 3D topology plot"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found!")
        print("\nTrying alternative locations...")
        
        # Try alternative paths
        alternatives = [
            Path("config.yaml"),
            Path("cofig.yaml"),  # Common typo in the repo
            Path("out/config.yaml")
        ]
        
        for alt_path in alternatives:
            if alt_path.exists():
                print(f"Found config at: {alt_path}")
                config_path = alt_path
                break
        else:
            print("Could not find config file!")
            return
    
    print(f"\nLoading configuration from: {config_path}")
    config = load_config(config_path)
    nodes = extract_nodes(config)
    
    if not nodes:
        print("Error: No nodes found in config file!")
        return
    
    # Print statistics
    print_statistics(nodes)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir.resolve()}")
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING TOPOLOGY PLOTS")
    print("="*60 + "\n")
    
    # Basic topology plot
    plot_topology(
        nodes,
        output_file=str(output_dir / "network_topology.png"),
        show_labels=not args.no_labels
    )
    
    # Topology with connections
    plot_topology_with_connections(
        nodes,
        output_file=str(output_dir / "network_topology_with_connections.png"),
        max_range=args.range,
        show_labels=not args.no_labels
    )
    
    # 3D plot (optional)
    if args.plot_3d:
        plot_topology_3d(
            nodes,
            output_file=str(output_dir / "network_topology_3d.png"),
            show_labels=not args.no_labels
        )
    
    print("\n" + "="*60)
    print("PLOTS COMPLETE!")
    print("="*60)
    print(f"\nAll plots saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
