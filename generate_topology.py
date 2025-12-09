"""
Random Network Topology Generator for Meshtastic Simulator

This script generates random node positions for network topology testing.
Control Center is always placed at (0, 0), while other nodes are randomly
distributed within specified coordinate bounds.

Usage:
    python generate_topology.py --dm 11 --sensor 11 --router 2 --x-low -1500 --x-high 2000 --y-low -1300 --y-high 1300
    python generate_topology.py --dm 10 --sensor 15 --router 3 --output custom_config.yaml
"""

import argparse
import random
import yaml
from pathlib import Path


def generate_random_topology(num_dm, num_sensor, num_router, 
                            x_low=-2000, x_high=2000, 
                            y_low=-2000, y_high=2000,
                            z_default=1.0, router_z=2.0,
                            min_distance=50.0,
                            max_distance=None,
                            seed=None):
    """
    Generate random node positions for network topology.
    
    Args:
        num_dm: Number of DM (Decision Maker) nodes
        num_sensor: Number of Sensor nodes
        num_router: Number of Router nodes
        x_low: Minimum x coordinate
        x_high: Maximum x coordinate
        y_low: Minimum y coordinate
        y_high: Maximum y coordinate
        z_default: Default z coordinate for non-router nodes
        router_z: Z coordinate for router nodes (elevated)
        min_distance: Minimum distance between nodes (to avoid overlap)
        max_distance: Maximum distance from Control Center (0,0). None means no limit.
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing node configurations in YAML format
    """
    if seed is not None:
        random.seed(seed)
    
    nodes = {}
    positions = []  # Track positions to enforce minimum distance
    node_id = 0
    
    # Helper function to generate a position with minimum/maximum distance check
    def generate_position(z_coord, max_attempts=1000):
        for _ in range(max_attempts):
            x = round(random.uniform(x_low, x_high), 1)
            y = round(random.uniform(y_low, y_high), 1)
            
            # Check maximum distance from Control Center (0, 0)
            if max_distance is not None:
                distance_from_center = (x**2 + y**2)**0.5
                if distance_from_center > max_distance:
                    continue
            
            # Check minimum distance from existing nodes
            if min_distance > 0:
                too_close = False
                for px, py in positions:
                    distance = ((x - px)**2 + (y - py)**2)**0.5
                    if distance < min_distance:
                        too_close = True
                        break
                
                if too_close:
                    continue
            
            positions.append((x, y))
            return x, y, z_coord
        
        # If we couldn't find a valid position, just use the last generated one
        print(f"Warning: Could not find position with constraints (min_distance={min_distance}m, max_distance={max_distance}m)")
        positions.append((x, y))
        return x, y, z_coord
    
    # Node 0: Control Center at origin (0, 0)
    nodes[node_id] = {
        'x': 0.0,
        'y': 0.0,
        'z': z_default,
        'simRole': 'Control_Center',
        'antennaGain': 0.0,
        'hopLimit': 5,
        'isClientMute': False,
        'isRepeater': False,
        'isRouter': False,
        'neighborInfo': False
    }
    positions.append((0.0, 0.0))
    node_id += 1
    
    # Generate DM nodes
    for i in range(num_dm):
        x, y, z = generate_position(z_default)
        nodes[node_id] = {
            'x': x,
            'y': y,
            'z': z,
            'simRole': 'DM',
            'antennaGain': 0.0,
            'hopLimit': 3,
            'isClientMute': False,
            'isRepeater': False,
            'isRouter': False,
            'neighborInfo': False
        }
        node_id += 1
    
    # Generate Sensor nodes
    for i in range(num_sensor):
        x, y, z = generate_position(z_default)
        nodes[node_id] = {
            'x': x,
            'y': y,
            'z': z,
            'simRole': 'Sensor',
            'antennaGain': 0.0,
            'hopLimit': 3,
            'isClientMute': False,
            'isRepeater': False,
            'isRouter': False,
            'neighborInfo': False
        }
        node_id += 1
    
    # Generate Router nodes (elevated z coordinate)
    for i in range(num_router):
        x, y, z = generate_position(router_z)
        nodes[node_id] = {
            'x': x,
            'y': y,
            'z': z,
            'simRole': 'Router',
            'antennaGain': 0.0,
            'hopLimit': 3,
            'isClientMute': False,
            'isRepeater': False,
            'isRouter': True,
            'neighborInfo': False
        }
        node_id += 1
    
    return nodes


def save_config(nodes, output_path):
    """
    Save node configuration to YAML file.
    
    Args:
        nodes: Dictionary of node configurations
        output_path: Path to output YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(nodes, f, default_flow_style=False, sort_keys=False)
    
    print(f"[OK] Configuration saved to: {output_path}")


def print_statistics(nodes):
    """
    Print statistics about the generated topology.
    
    Args:
        nodes: Dictionary of node configurations
    """
    # Count nodes by type
    node_counts = {}
    for node_id, node in nodes.items():
        role = node['simRole']
        node_counts[role] = node_counts.get(role, 0) + 1
    
    # Calculate bounds
    x_coords = [node['x'] for node in nodes.values()]
    y_coords = [node['y'] for node in nodes.values()]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    print("\n" + "="*60)
    print("TOPOLOGY GENERATION SUMMARY")
    print("="*60)
    print(f"\nTotal Nodes: {len(nodes)}")
    print("\nNode Distribution:")
    for role, count in sorted(node_counts.items()):
        print(f"  {role:20s}: {count:3d} nodes")
    
    print(f"\nNetwork Dimensions:")
    print(f"  X range: [{x_min:.1f}, {x_max:.1f}] m (width: {width:.1f} m)")
    print(f"  Y range: [{y_min:.1f}, {y_max:.1f}] m (height: {height:.1f} m)")
    print(f"  Area: {width * height / 1e6:.2f} km²")
    
    # Calculate average distance from Control Center
    control_center = nodes[0]
    distances = []
    for node_id, node in nodes.items():
        if node_id == 0:
            continue
        dx = node['x'] - control_center['x']
        dy = node['y'] - control_center['y']
        distance = (dx**2 + dy**2)**0.5
        distances.append(distance)
    
    if distances:
        avg_distance = sum(distances) / len(distances)
        max_distance = max(distances)
        print(f"\nDistance from Control Center (0, 0):")
        print(f"  Average: {avg_distance:.1f} m")
        print(f"  Maximum: {max_distance:.1f} m")
    
    print("="*60 + "\n")


def load_params_from_yaml(yaml_path):
    """
    Load generation parameters from YAML file.
    
    Args:
        yaml_path: Path to YAML parameter file
    
    Returns:
        Dictionary of parameters
    """
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def main():
    parser = argparse.ArgumentParser(
        description='Generate random network topology for Meshtastic simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default topology (11 DM, 11 Sensor, 2 Router)
  python generate_topology.py
  
  # Generate from YAML parameter file
  python generate_topology.py --input-params topology_params.yaml
  
  # Generate custom topology with specific bounds
  python generate_topology.py --dm 15 --sensor 20 --router 3 --x-low -3000 --x-high 3000
  
  # Generate topology with custom output file
  python generate_topology.py --dm 10 --sensor 10 --router 2 --output custom_topology.yaml
  
  # Generate reproducible topology with seed
  python generate_topology.py --dm 10 --sensor 15 --seed 42
        """
    )
    
    # Input parameter file
    parser.add_argument('--input-params', '--input', '-i', type=str, default=None,
                       help='YAML file with generation parameters (overrides other arguments)')
    
    # Node counts
    parser.add_argument('--dm', type=int, default=11,
                       help='Number of DM (Decision Maker) nodes (default: 11)')
    parser.add_argument('--sensor', type=int, default=11,
                       help='Number of Sensor nodes (default: 11)')
    parser.add_argument('--router', type=int, default=2,
                       help='Number of Router nodes (default: 2)')
    
    # Coordinate bounds
    parser.add_argument('--x-low', type=float, default=-2000,
                       help='Minimum x coordinate in meters (default: -2000)')
    parser.add_argument('--x-high', type=float, default=2000,
                       help='Maximum x coordinate in meters (default: 2000)')
    parser.add_argument('--y-low', type=float, default=-2000,
                       help='Minimum y coordinate in meters (default: -2000)')
    parser.add_argument('--y-high', type=float, default=2000,
                       help='Maximum y coordinate in meters (default: 2000)')
    
    # Other parameters
    parser.add_argument('--z-default', type=float, default=1.0,
                       help='Z coordinate for non-router nodes (default: 1.0)')
    parser.add_argument('--router-z', type=float, default=2.0,
                       help='Z coordinate for router nodes (default: 2.0)')
    parser.add_argument('--min-distance', type=float, default=50.0,
                       help='Minimum distance between nodes in meters (default: 50)')
    parser.add_argument('--max-distance', type=float, default=None,
                       help='Maximum distance from Control Center (0,0) in meters (default: None - no limit)')
    
    # Output and reproducibility
    parser.add_argument('--output', '-o', type=str, default='out/config.yaml',
                       help='Output YAML file path (default: out/config.yaml)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (default: None)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress statistics output')
    
    args = parser.parse_args()
    
    # Load parameters from YAML if provided
    if args.input_params:
        print(f"Loading parameters from: {args.input_params}")
        params = load_params_from_yaml(args.input_params)
        
        # Save the command-line output argument (it should take precedence)
        cli_output = args.output if args.output != 'out/config.yaml' else None
        cli_seed = args.seed
        cli_quiet = args.quiet
        
        # Override args with YAML values
        args.dm = params.get('dm', args.dm)
        args.sensor = params.get('sensor', args.sensor)
        args.router = params.get('router', args.router)
        args.x_low = params.get('x_low', args.x_low)
        args.x_high = params.get('x_high', args.x_high)
        args.y_low = params.get('y_low', args.y_low)
        args.y_high = params.get('y_high', args.y_high)
        args.z_default = params.get('z_default', args.z_default)
        args.router_z = params.get('router_z', args.router_z)
        args.min_distance = params.get('min_distance', args.min_distance)
        args.max_distance = params.get('max_distance', args.max_distance)
        
        # Only override these if they weren't explicitly set on command line
        if cli_seed is not None:
            args.seed = cli_seed
        else:
            args.seed = params.get('seed', args.seed)
        
        if cli_output is not None:
            args.output = cli_output
        else:
            args.output = params.get('output', args.output)
        
        args.quiet = params.get('quiet', cli_quiet)
    
    # Validate inputs
    if args.dm < 0 or args.sensor < 0 or args.router < 0:
        parser.error("Node counts must be non-negative")
    
    if args.x_low >= args.x_high or args.y_low >= args.y_high:
        parser.error("Invalid coordinate bounds: low must be less than high")
    
    # Generate topology
    print(f"\nGenerating topology with {args.dm} DM, {args.sensor} Sensor, {args.router} Router nodes...")
    print(f"Coordinate bounds: X[{args.x_low}, {args.x_high}], Y[{args.y_low}, {args.y_high}]")
    if args.max_distance is not None:
        print(f"Maximum distance from center: {args.max_distance}m")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    
    nodes = generate_random_topology(
        num_dm=args.dm,
        num_sensor=args.sensor,
        num_router=args.router,
        x_low=args.x_low,
        x_high=args.x_high,
        y_low=args.y_low,
        y_high=args.y_high,
        z_default=args.z_default,
        router_z=args.router_z,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        seed=args.seed
    )
    
    # Save configuration
    save_config(nodes, args.output)
    
    # Print statistics
    if not args.quiet:
        print_statistics(nodes)
    
    print(f"[OK] Ready to simulate! Use this config with:")
    print(f"  python loraMesh.py --config {args.output}")


if __name__ == '__main__':
    main()
