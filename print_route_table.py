#!/usr/bin/env python3
# filepath: /home/raveen/Final_Year_Project/Simulator_new/Meshtasticator/get_route_table.py
"""
Script to request and print the stored AODV routing table from a Meshtastic node.
"""
from meshtastic.tcp_interface import TCPInterface
import sys
import os
import time
from pubsub import pub

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'meshtastic'))
import aodv_pb2

AODV_ROUTING_APP_PORTNUM = 75

class RouteTablePrinter:
    def __init__(self):
        self.route_table = None
        self.request_id = int(time.time())
        
    def on_receive(self, packet, interface):
        """Callback to handle route table response"""
        try:
            if 'decoded' not in packet:
                return
                
            decoded = packet['decoded']
            portnum = decoded.get('portnum')
            
            if portnum == 'AODV_ROUTING_APP' or portnum == AODV_ROUTING_APP_PORTNUM:
                aodv_msg = aodv_pb2.AODV()
                aodv_msg.ParseFromString(decoded['payload'])
                
                # Check if it's a route table response
                if aodv_msg.HasField('rt_response'):
                    rt_resp = aodv_msg.rt_response
                    if rt_resp.request_id == self.request_id:
                        self.route_table = rt_resp.routes
                        print("✓ Route table received!")
                        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def print_routes(self):
        """Print the routing table"""
        if not self.route_table:
            print("No route table data received.")
            return
            
        print("\n" + "="*80)
        print("AODV ROUTING TABLE")
        print("="*80)
        print(f"{'Destination':<15} {'Next Hop':<15} {'Hops':<8} {'Seq Num':<10} "
              f"{'Lifetime':<10} {'Valid':<8}")
        print("-"*80)
        
        valid_routes = 0
        for route in self.route_table:
            dest = f"!{route.destination:08x}"
            next_hop = f"!{route.next_hop:08x}"
            valid_str = "✓" if route.valid else "✗"
            
            print(f"{dest:<15} {next_hop:<15} {route.hop_count:<8} "
                  f"{route.destination_seq_num:<10} {route.lifetime:<10} {valid_str:<8}")
            
            if route.valid:
                valid_routes += 1
        
        print("="*80)
        print(f"Total routes: {len(self.route_table)} | Valid routes: {valid_routes}\n")

def get_route_table(node_port=4403, timeout=10):
    """
    Request and print the routing table from a node.
    
    Args:
        node_port: TCP port of the node
        timeout: How long to wait for response (seconds)
    """
    print(f"Connecting to node on port {node_port}...")
    
    printer = RouteTablePrinter()
    pub.subscribe(printer.on_receive, "meshtastic.receive")
    
    iface = TCPInterface(hostname="127.0.0.1", portNumber=node_port)
    
    try:
        # Get node info
        node_info = iface.getMyNodeInfo()
        node_id = node_info.get('user', {}).get('id', 'Unknown')
        node_num = node_info.get('num', 'Unknown')
        
        print(f"✓ Connected to Node: {node_id} (Num: !{node_num:08x})")
        
        # Create route table request
        print("Requesting routing table...")
        msg = aodv_pb2.AODV()
        msg.rt_request.request_id = printer.request_id
        
        payload = msg.SerializeToString()
        
        # Send request to self (local node)
        iface.sendData(
            data=payload,
            destinationId=f"!{node_num:08x}",
            portNum=AODV_ROUTING_APP_PORTNUM,
            wantAck=False,
            channelIndex=0,
        )
        
        # Wait for response
        print(f"Waiting for response (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if printer.route_table is not None:
                break
            time.sleep(0.5)
        
        if printer.route_table is None:
            print("\n✗ No response received. Possible issues:")
            print("  1. Firmware doesn't support route table requests")
            print("  2. Node has no routes stored")
            print("  3. Message handler not implemented")
            print("\nYou need to implement rt_request handler in firmware!")
        else:
            printer.print_routes()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        iface.close()
        print("Connection closed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get AODV routing table from Meshtastic node")
    parser.add_argument("-p", "--port", type=int, default=4403,
                        help="TCP port of the node (default: 4403)")
    parser.add_argument("-t", "--timeout", type=int, default=10,
                        help="Response timeout in seconds (default: 10)")
    
    args = parser.parse_args()
    
    get_route_table(args.port, args.timeout)