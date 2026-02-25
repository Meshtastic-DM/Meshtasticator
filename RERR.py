#!/usr/bin/env python3
# send_rerr_tcp.py
from meshtastic.tcp_interface import TCPInterface
# Generated from your AODV .proto (you need to generate this file)
import aodv_pb2

# Use your enum value for meshtastic_PortNum_AODV_ROUTING_APP
AODV_ROUTING_APP_PORTNUM = 75  # <-- replace with your actual value

def main():
    iface = TCPInterface(hostname="127.0.0.1", portNumber=4403)
    try:
        msg = aodv_pb2.AODV()
        # Build RERR payload per your protobuf definition
        msg.rerr.unreachable_destinations.add(node_num=0x00000012, seq_num=7)

        payload = msg.SerializeToString()

        iface.sendData(
            data=payload,
            destinationId="!00000012",      # or "!xxxxxxxx" for unicast
            portNum=AODV_ROUTING_APP_PORTNUM,
            wantAck=False,
            channelIndex=0,
        )
        print("RERR sent")
    finally:
        iface.close()

if __name__ == "__main__":
    main()
