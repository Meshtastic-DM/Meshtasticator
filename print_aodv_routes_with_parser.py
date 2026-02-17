#!/usr/bin/env python3
"""
Advanced script to retrieve and print the AODV routing table with proper response parsing.
"""
import sys
import os
import time
import threading

# Add the protobufs directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'protobufs'))

try:
    from meshtastic.tcp_interface import TCPInterface
    from meshtastic import portnums_pb2
except ImportError:
    print("Error: meshtastic Python package not found. Install with: pip install meshtastic")
    sys.exit(1)

try:
    from meshtastic import aodv_pb2
except ImportError:
    print("Error: aodv_pb2.py not found. You need to generate Python protobuf files.")
    print("\nTo generate the file, run:")
    print("  cd protobufs")
    print("  python -m grpc_tools.protoc -I. --python_out=. meshtastic/aodv.proto")
    print("\nOr if you have protoc installed:")
    print("  cd protobufs && protoc --python_out=. meshtastic/aodv.proto")
    print("\nYou may need to install grpcio-tools: pip install grpcio-tools")
    sys.exit(1)

# Port number for AODV routing application
AODV_ROUTING_APP_PORTNUM = 75

class RouteTableCollector:
    """Collects route table responses from nodes."""
    
    def __init__(self):
        self.route_tables = {}
        self.received_event = threading.Event()
        
    def onReceive(self, packet, interface):
        """Callback for received packets."""
        try:
            # Check if this is an AODV routing packet
            if 'decoded' in packet and 'portnum' in packet['decoded']:
                portnum = packet['decoded']['portnum']
                
                # Check for AODV_ROUTING_APP
                if portnum == 'AODV_ROUTING_APP' or portnum == AODV_ROUTING_APP_PORTNUM:
                    payload = packet['decoded'].get('payload')
                    
                    if payload:
                        # Parse AODV message
                        aodv_msg = aodv_pb2.AODV()
                        aodv_msg.ParseFromString(payload)
                        
                        # Check if it's a route table response
                        if aodv_msg.HasField('rt_resp'):
                            rt_resp = aodv_msg.rt_resp
                            node_id = rt_resp.node_id
                            
                            # Store the route table
                            self.route_tables[node_id] = rt_resp
                            self.received_event.set()
                            
                            print(f"\n✓ Received route table from node 0x{node_id:08x}")
                            
        except Exception as e:
            print(f"Error parsing packet: {e}")

def print_route_table(node_port=4403, timeout=5):
    """
    Connect to a Meshtastic node and request/print its routing table.
    
    Args:
        node_port: TCP port of the target node (default: 4403)
        timeout: Seconds to wait for response (default: 5)
    """
    print(f"Connecting to node on port {node_port}...")
    
    try:
        iface = TCPInterface(hostname="127.0.0.1", portNumber=node_port)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return None
    
    collector = RouteTableCollector()
    
    try:
        # Subscribe to receive callbacks
        pub.subscribe(collector.onReceive, "meshtastic.receive")
        
        # Get the node's own ID
        my_node_num = iface.myInfo.my_node_num
        
        # Create AODV RouteTableRequest message
        msg = aodv_pb2.AODV()
        msg.rt_req.CopyFrom(aodv_pb2.RouteTableRequest())
        
        payload = msg.SerializeToString()
        
        print(f"Requesting routing table from node 0x{my_node_num:08x}...")
        
        # Send request to the node itself
        iface.sendData(
            data=payload,
            destinationId=my_node_num,
            portNum=AODV_ROUTING_APP_PORTNUM,
            wantAck=False,
            channelIndex=0,
        )
        
        # Wait for response
        print(f"Waiting up to {timeout} seconds for response...")
        if collector.received_event.wait(timeout=timeout):
            print("\n" + "="*70)
            print(f"ROUTING TABLE FOR NODE 0x{my_node_num:08x}")
            print("="*70)
            
            rt_resp = collector.route_tables.get(my_node_num)
            if rt_resp:
                print(f"Node Sequence Number: {rt_resp.node_seq_num}")
                print(f"Number of Routes: {len(rt_resp.routes)}")
                print("-"*70)
                
                if rt_resp.routes:
                    print(f"{'Destination':<18} {'Next Hop':<15} {'Hops':<6} {'Seq Num':<10} {'Expiry (s)':<10}")
                    print("-"*70)
                    
                    for route in rt_resp.routes:
                        print(f"0x{route.destination:08x}       "
                              f"0x{route.next_hop:02x}           "
                              f"{route.hop_count:<6} "
                              f"{route.dest_seq_num:<10} "
                              f"{route.expiry_seconds:<10}")
                else:
                    print("No active routes found.")
                    
                print("="*70 + "\n")
                return rt_resp
        else:
            print("✗ Timeout waiting for route table response.")
            print("  Make sure the AODV module is enabled in the firmware.")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        iface.close()
        print("Connection closed.\n")

def print_all_nodes_routes(start_port=4403, num_nodes=5):
    """
    Print routing tables for multiple nodes in the simulator.
    
    Args:
        start_port: First node's TCP port
        num_nodes: Number of nodes to query
    """
    all_route_tables = {}
    
    for i in range(num_nodes):
        port = start_port + i
        print(f"\n{'#'*70}")
        print(f"QUERYING NODE {i} (Port {port})")
        print(f"{'#'*70}")
        
        try:
            rt_resp = print_route_table(port, timeout=5)
            if rt_resp:
                all_route_tables[port] = rt_resp
        except Exception as e:
            print(f"Failed to query node {i}: {e}")
        
        time.sleep(1)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Successfully queried {len(all_route_tables)} out of {num_nodes} nodes")
    print("="*70 + "\n")

if __name__ == "__main__":
    import argparse
    
    # Import pubsub for receive callbacks
    try:
        from pubsub import pub
    except ImportError:
        print("Error: pypubsub not found. Install with: pip install pypubsub")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="Print Meshtastic AODV routing tables")
    parser.add_argument("-p", "--port", type=int, default=4403,
                        help="TCP port of the node (default: 4403)")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Print routing tables for all nodes in simulator")
    parser.add_argument("-n", "--num-nodes", type=int, default=5,
                        help="Number of nodes to query when using --all (default: 5)")
    parser.add_argument("-t", "--timeout", type=int, default=5,
                        help="Timeout in seconds for each query (default: 5)")
    
    args = parser.parse_args()
    
    if args.all:
        print_all_nodes_routes(args.port, args.num_nodes)
    else:
        print_route_table(args.port, args.timeout)
