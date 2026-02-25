#!/usr/bin/env python3
# send_admin_message.py
from meshtastic.tcp_interface import TCPInterface
from meshtastic import admin_pb2, mesh_pb2

ADMIN_APP_PORTNUM = 6  # meshtastic_PortNum_ADMIN_APP

def send_get_owner_request(iface, destination_node_id):
    """Example: Send get_owner request"""
    admin_msg = admin_pb2.AdminMessage()
    admin_msg.get_owner_request = True
    
    payload = admin_msg.SerializeToString()
    
    iface.sendData(
        data=payload,
        destinationId=destination_node_id,  # e.g., "!00000010" or "^all" for broadcast
        portNum=ADMIN_APP_PORTNUM,
        wantAck=True,
        channelIndex=0,
    )
    print(f"Admin get_owner_request sent to {destination_node_id}")


def send_set_owner(iface, destination_node_id, long_name, short_name):
    """Example: Send set_owner command"""
    admin_msg = admin_pb2.AdminMessage()
    admin_msg.set_owner.long_name = long_name
    admin_msg.set_owner.short_name = short_name
    
    payload = admin_msg.SerializeToString()
    
    iface.sendData(
        data=payload,
        destinationId=destination_node_id,
        portNum=ADMIN_APP_PORTNUM,
        wantAck=True,
        channelIndex=0,
    )
    print(f"Admin set_owner sent to {destination_node_id}")


def send_get_config_request(iface, destination_node_id, config_type):
    """Example: Send get_config_request
    config_type options:
    - admin_pb2.AdminMessage.DEVICE_CONFIG
    - admin_pb2.AdminMessage.POSITION_CONFIG
    - admin_pb2.AdminMessage.LORA_CONFIG
    - admin_pb2.AdminMessage.SECURITY_CONFIG
    etc.
    """
    admin_msg = admin_pb2.AdminMessage()
    admin_msg.get_config_request = config_type
    
    payload = admin_msg.SerializeToString()
    
    iface.sendData(
        data=payload,
        destinationId=destination_node_id,
        portNum=ADMIN_APP_PORTNUM,
        wantAck=True,
        channelIndex=0,
    )
    print(f"Admin get_config_request ({config_type}) sent to {destination_node_id}")


def send_reboot(iface, destination_node_id, seconds=5):
    """Example: Send reboot command"""
    admin_msg = admin_pb2.AdminMessage()
    admin_msg.reboot_seconds = seconds
    
    payload = admin_msg.SerializeToString()
    
    iface.sendData(
        data=payload,
        destinationId=destination_node_id,
        portNum=ADMIN_APP_PORTNUM,
        wantAck=True,
        channelIndex=0,
    )
    print(f"Admin reboot ({seconds}s) sent to {destination_node_id}")


def main():
    # Connect to node on localhost:4403
    iface = TCPInterface(hostname="127.0.0.1", portNumber=4403)
    
    try:
        # Target node (SDN controller node or any other node)
        target_node = "!00000012"  # Replace with your target node ID
        
        # Example 1: Get owner info
        send_get_owner_request(iface, target_node)
        
        # Example 2: Set owner (change node name)
        # send_set_owner(iface, target_node, "Test Node", "TST")
        
        # Example 3: Get security config (to see admin keys)
        # send_get_config_request(iface, target_node, admin_pb2.AdminMessage.SECURITY_CONFIG)
        
        # Example 4: Reboot node
        # send_reboot(iface, target_node, seconds=10)
        
        # Wait a bit for responses (optional)
        import time
        time.sleep(2)
        
    finally:
        iface.close()


if __name__ == "__main__":
    main()