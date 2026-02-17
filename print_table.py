#!/usr/bin/env python3
"""
Script to retrieve and print the AODV routing table from a Meshtastic node via TCP interface.
"""
from meshtastic.tcp_interface import TCPInterface
import aodv_pb2
import time
import sys

# Port number for AODV routing application
AODV_ROUTING_APP_PORTNUM = 75

def on_receive(packet, interface):
    """Callback to handle received packets"""
    try:
        if 'decoded' in packet and packet['decoded'].get('portnum') == 'AODV_ROUTING_APP':
            print(f"\nReceived AODV packet: {packet}")
            # Parse AODV message
            aodv_msg = aodv_pb2.AODV()
            aodv_msg.ParseFromString(packet['decoded']['payload'])
            print(f"AODV Message: {aodv_msg}")
    except Exception as e:
        print(f"Error parsing packet: {e}")

def print_route_table(node_port=4403):
    """
    Connect to a Meshtastic node and print its routing table.
    
    Args:
        node_port: TCP port of the target node (default: 4403)
    """
    print(f"Connecting to node on port {node_port}...")
    iface = TCPInterface(hostname="127.0.0.1", portNumber=node_port)
    
    try:
        # Subscribe to receive AODV messages
        pub.subscribe(on_receive, "meshtastic.receive")
        
        print("\n" + "="*60)
        print(f"ROUTING TABLE FOR NODE ON PORT {node_port}")
        print("="*60)
        print(f"{'Destination':<15} {'Next Hop':<15} {'Hop Count':<10} {'Seq Num':<10}")
        print("-"*60)
        
        # Get node info
        node_info = iface.getMyNodeInfo()
        print(f"Node ID: {node_info.get('user', {}).get('id', 'Unknown')}")
        print(f"Node Num: {node_info.get('num', 'Unknown')}")
        
        # Listen for AODV messages for a few seconds
        print("\nListening for AODV routing messages...")
        time.sleep(5)
        
        print("\nNote: Direct route table access requires firmware support.")
        print("Consider adding a custom admin message to export routing table.")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        iface.close()
        print("Connection closed.")

def send_route_table_request(node_port=4403):
    """
    Send a simple AODV message to trigger routing activity.
    """
    print(f"Connecting to node on port {node_port}...")
    iface = TCPInterface(hostname="127.0.0.1", portNumber=node_port)
    
    try:
        # Create a simple AODV message based on your actual protobuf structure
        msg = aodv_pb2.AODV()
        
        # Check what fields are available in your AODV message
        print("Available AODV message types:")
        print(msg.DESCRIPTOR.fields_by_name.keys())
        
        # You likely have: rreq, rrep, rerr, etc.
        # For now, just print the structure
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        iface.close()

def print_all_nodes_routes(start_port=4403, num_nodes=5):
    """
    Print routing tables for multiple nodes in the simulator.
    
    Args:
        start_port: First node's TCP port
        num_nodes: Number of nodes to query
    """
    for i in range(num_nodes):
        port = start_port + i
        print(f"\n{'#'*60}")
        print(f"NODE {i} (Port {port})")
        print(f"{'#'*60}")
        try:
            print_route_table(port)
        except Exception as e:
            print(f"Failed to connect to node {i}: {e}")
        time.sleep(1)

if __name__ == "__main__":
    import argparse
    from pubsub import pub
    
    parser = argparse.ArgumentParser(description="Print Meshtastic AODV routing tables")
    parser.add_argument("-p", "--port", type=int, default=4403,
                        help="TCP port of the node (default: 4403)")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Print routing tables for all nodes in simulator")
    parser.add_argument("-n", "--num-nodes", type=int, default=5,
                        help="Number of nodes to query when using --all (default: 5)")
    parser.add_argument("--check-proto", action="store_true",
                        help="Check AODV protobuf structure")
    
    args = parser.parse_args()
    
    if args.check_proto:
        send_route_table_request(args.port)
    elif args.all:
        print_all_nodes_routes(args.port, args.num_nodes)
    else:
        print_route_table(args.port)