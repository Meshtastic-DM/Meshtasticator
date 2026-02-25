#!/usr/bin/env python3
# send_sdn_route_tcp.py

from meshtastic.tcp_interface import TCPInterface
from generated import sdn_pb2, portnums_pb2

def main():
    # Connect to meshtasticd TCP (same as your RERR script)
    iface = TCPInterface(hostname="127.0.0.1", portNumber=4403)

    try:
        # Build SDN Route Command
        route_cmd = sdn_pb2.SDNRouteCommand()
        route_cmd.destination = 0x00000015      # Replace with actual destination
        route_cmd.next_hop = 0x12         # Replace with backup next hop

        # Wrap inside SDN message
        sdn_msg = sdn_pb2.SDN()
        sdn_msg.route_command.CopyFrom(route_cmd)

        payload = sdn_msg.SerializeToString()

        # Send over TCP
        iface.sendData(
            data=payload,
            destinationId="!00000011",   # Target node (string format for TCP)
            portNum=portnums_pb2.PortNum.SDN_APP,
            wantAck=False,
            channelIndex=0,
        )

        print("SDN Route Command sent via TCP")

    finally:
        iface.close()

if __name__ == "__main__":
    main()