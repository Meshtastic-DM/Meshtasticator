#!/usr/bin/env python3
# send_sdn_announcement_tcp.py

from meshtastic.tcp_interface import TCPInterface
from generated import sdn_pb2, portnums_pb2
import time
import hmac
import hashlib
import struct
import base64

def build_hmac_message(controller_id, seq_num, timestamp, public_key):
    """Build the HMAC input message (44 bytes): controller_id + seq + timestamp + pubkey"""
    msg = struct.pack('<I', controller_id)  # 4 bytes little-endian
    msg += struct.pack('<I', seq_num)       # 4 bytes little-endian
    msg += struct.pack('<I', timestamp)     # 4 bytes little-endian
    msg += public_key                        # 32 bytes
    return msg

def calculate_hmac_sha256_16(secret, message):
    """Calculate HMAC-SHA256 and truncate to first 16 bytes"""
    h = hmac.new(secret.encode('utf-8'), message, hashlib.sha256)
    return h.digest()[:16]  # Truncate to 16 bytes

def main():
    # Connect to meshtasticd TCP
    iface = TCPInterface(hostname="127.0.0.1", portNumber=4403)
    
    try:
        # Get our node info to use as controller
        node_info = iface.getMyNodeInfo()
        controller_id = node_info['num']  # Our node number
        
        # Configuration (must match SDNModule.cpp)
        HMAC_SECRET = "meshtastic-sdn-secret"
        
        # Use actual controller's Curve25519 public key from Node 0
        public_key = base64.b64decode("gO/NerZ7JaKbrvHgaKwL8nhPB4W9ZxJTzYUIcbm5Iho=")
        
        # Build SDN Announcement
        announcement = sdn_pb2.SDNAnnouncement()
        announcement.sequence_num = 1  # Increment for each announcement
        announcement.timestamp = int(time.time())
        announcement.public_key = public_key
        
        # Calculate HMAC over: controller_id + seq + timestamp + public_key
        hmac_msg = build_hmac_message(
            controller_id,
            announcement.sequence_num,
            announcement.timestamp,
            public_key
        )
        announcement.hmac_hash = calculate_hmac_sha256_16(HMAC_SECRET, hmac_msg)
        
        # Wrap inside SDN message
        sdn_msg = sdn_pb2.SDN()
        sdn_msg.announcement.CopyFrom(announcement)
        
        payload = sdn_msg.SerializeToString()
        
        # Send broadcast
        iface.sendData(
            data=payload,
            destinationId="^all",  # Broadcast to all nodes
            portNum=portnums_pb2.PortNum.SDN_APP,
            wantAck=False,
            channelIndex=0,
            hopLimit=7  # Match SDNModule broadcast hop limit
        )
        
        print(f"SDN Announcement broadcast sent:")
        print(f"  Controller: 0x{controller_id:08x}")
        print(f"  Sequence: {announcement.sequence_num}")
        print(f"  Timestamp: {announcement.timestamp}")
        print(f"  HMAC: {announcement.hmac_hash.hex()}")
        
    finally:
        iface.close()

if __name__ == "__main__":
    main()