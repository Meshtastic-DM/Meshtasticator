#!/usr/bin/env python3
"""
Script to retrieve and print the AODV routing table from a Meshtastic node via TCP interface.
"""
import sys
import os
import time

# Add the protobufs directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'protobufs'))

try:
    from meshtastic.tcp_interface import TCPInterface
except ImportError:
    print("Error: meshtastic Python package not found. Install with: pip install meshtastic")
    sys.exit(1)

try:
    from meshtastic import aodv_pb2
except ImportError:
    print("Error: aodv_pb2.py not found. You need to generate Python protobuf files.")
    print("Run: cd protobufs && python -m grpc_tools.protoc -I. --python_out=. meshtastic/aodv.proto")
    print("Or install protobuf compiler and run: protoc --python_out=. meshtastic/aodv.proto")
    sys.exit(1)

# Port number for AODV routing application
AODV_ROUTING_APP_PORTNUM = 75

def print_route_table(node_port=4403):
    """
    Connect to a Meshtastic node and request/print its routing table.
    
    Args:
        node_port: TCP port of the target node (default: 4403)
    """
    print(f"Connecting to node on port {node_port}...")
    
    try:
        iface = TCPInterface(hostname="127.0.0.1", portNumber=node_port)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return
    
    try:
        # Create AODV RouteTableRequest message
        msg = aodv_pb2.AODV()
        msg.rt_req.CopyFrom(aodv_pb2.RouteTableRequest())  # Send route table query
        
        payload = msg.SerializeToString()
        
        print("Requesting routing table...")
        
        # Send request to the node itself
        iface.sendData(
            data=payload,
            destinationId=iface.myInfo.my_node_num,  # Send to self to trigger response
            portNum=AODV_ROUTING_APP_PORTNUM,
            wantAck=False,
            channelIndex=0,
        )
        
        # Wait for response
        print("Waiting for response...")
        time.sleep(3)
        
        # The response will be logged by the firmware
        # For programmatic access, you would need to add a packet handler
        print("\nNote: Route table response is sent back via AODV protocol.")
        print("Check the firmware logs or add a packet receive handler to parse the response.")
        print("\nTo see the response, you can add this to your script:")
        print("  def onReceive(packet, interface):")
        print("      if packet['decoded']['portnum'] == 'AODV_ROUTING_APP':")
        print("          # Parse packet['decoded']['payload']")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        iface.close()
        print("\nConnection closed.")

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
    
    parser = argparse.ArgumentParser(description="Print Meshtastic AODV routing tables")
    parser.add_argument("-p", "--port", type=int, default=4403,
                        help="TCP port of the node (default: 4403)")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Print routing tables for all nodes in simulator")
    parser.add_argument("-n", "--num-nodes", type=int, default=5,
                        help="Number of nodes to query when using --all (default: 5)")
    
    args = parser.parse_args()
    
    if args.all:
        print_all_nodes_routes(args.port, args.num_nodes)
    else:
        print_route_table(args.port)
