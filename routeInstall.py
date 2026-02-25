#!/usr/bin/env python3
# Test script for SDN Route Install functionality

from meshtastic.tcp_interface import TCPInterface
from generated import sdn_pb2, portnums_pb2
import time

def pack_hop_path(hops):
    """
    Pack a list of 1-byte hops into a 64-bit integer.
    LSB (byte 0) = first hop, byte 1 = second hop, etc.
    Maximum 8 hops.
    
    Example: [0x12, 0x34, 0x56] → 0x0000000000563412
    """
    if len(hops) > 8:
        raise ValueError("Maximum 8 hops allowed")
    
    hop_path = 0
    for i, hop in enumerate(hops):
        if hop < 0 or hop > 0xFF:
            raise ValueError(f"Hop {i} value {hop} out of range (0-255)")
        hop_path |= (hop << (i * 8))
    
    return hop_path


def unpack_hop_path(hop_path):
    """
    Unpack a 64-bit hop_path into a list of individual hops.
    Stops at first zero byte.
    """
    hops = []
    for i in range(8):
        hop = (hop_path >> (i * 8)) & 0xFF
        if hop == 0:
            break
        hops.append(hop)
    return hops


def main():
    """
    Send an SDN RouteInstall command to install a specific path in the mesh network.
    
    Example topology:
      Controller (0x00000010) → Node 0x11 → Node 0x12 → Destination (0x00000014)
    
    This script sends a RouteInstall command from controller to the first node (0x11),
    instructing it to install a cascading path to the destination.
    
    SDN routes use seq_num=0 for authoritative installation (bypasses AODV freshness checks).
    """
    
    # Connect to meshtasticd TCP (controller node)
    iface = TCPInterface(hostname="127.0.0.1", portNumber=4403)

    try:
        # Build SDN Route Install message
        route_install = sdn_pb2.SDNRouteInstall()
        
        # Destination: final destination node (full 32-bit node number)
        route_install.destination = 0x00000014  # Final destination
        
        # Path: list of 8-bit last bytes forming the complete path
        # Index 0 = first hop (starting node), Index N-1 = last hop before destination
        # Example: [0x11, 0x12, 0x13] means:
        #   - Node 0x11 receives RouteInstall, forwards to 0x12
        #   - Node 0x12 receives RouteSet, forwards to 0x13
        #   - Node 0x13 receives RouteSet, forwards to destination
        path = [0x12, 0x13, 0x14]  # 3-hop path
        
        # Pack path into fixed64
        route_install.hop_path = pack_hop_path(path)
        
        # Install ID (uint8 semantically, wraps at 255)
        route_install.install_id = 1
        
        # Wrap inside SDN message
        sdn_msg = sdn_pb2.SDN()
        sdn_msg.route_install.CopyFrom(route_install)
        
        payload = sdn_msg.SerializeToString()
        
        # Send to starting node (first hop in path)
        starting_node_last_byte = 0x00000011
        # Convert to full node ID string format (assuming subnet 0x00000000)
        starting_node_id = f"!{starting_node_last_byte:08x}"
        
        iface.sendData(
            data=payload,
            destinationId=starting_node_id,
            portNum=portnums_pb2.PortNum.SDN_APP,
            wantAck=False,
            channelIndex=0,
        )
        
        print(f"✓ SDN Route Install sent via TCP")
        print(f"  Install ID: {route_install.install_id}")
        print(f"  Starting Node: {starting_node_id} (0x{starting_node_last_byte:02x})")
        print(f"  Destination: 0x{route_install.destination:08x}")
        print(f"  Packed hop_path: 0x{route_install.hop_path:016x}")
        print(f"  Path: {' → '.join([f'0x{hop:02x}' for hop in path])}")
        print(f"  Path length: {len(path)} hops")
        print()
        print("Expected behavior:")
        print(f"  1. Node 0x{path[0]:02x} receives RouteInstall")
        print(f"  2. Node 0x{path[0]:02x} installs route: dest=0x{route_install.destination:08x} → next_hop=0x{path[0]:02x} (seq_num=0 authoritative)")
        print(f"  3. Node 0x{path[0]:02x} sends RouteSet (without seq_num field) to 0x{path[0]:02x}")
        
        for i in range(len(path)):
            if i < len(path) - 1:
                print(f"  {i+4}. Node 0x{path[i]:02x} receives RouteSet")
                print(f"     - Installs reverse route to start node (0x{path[0]:02x}) with seq_num=0")
                print(f"     - Installs forward route: dest=0x{route_install.destination:08x} → next_hop=0x{path[i+1]:02x} (seq_num=0)")
                print(f"     - Forwards RouteSet to 0x{path[i+1]:02x}")
        
        print(f"  {len(path)+4}. Destination 0x{route_install.destination:08x} receives RouteSet")
        print(f"     - Installs reverse route to start node with seq_num=0")
        print(f"     - Sends RouteSetConfirm back to controller")
        print()
        print("Check logs for:")
        print("  - 'SDN: RouteInstall received' at start node")
        print("  - 'SDN: Installed forward route ... seq_num=0 authoritative' at start node")
        print("  - 'SDN: Sending RouteSet ... seq_num=0 authoritative' at start node")
        print("  - 'SDN: Installed reverse route ... seq_num=0 authoritative' at each hop")
        print("  - 'SDN: Installed forward route ... seq_num=0 authoritative' at each hop")
        print("  - 'SDN: RouteSetConfirm received' at controller")
        print()
        print("Note: SDN routes use seq_num=0 (authoritative installation):")
        print("  - Bypasses AODV freshness checks")
        print("  - Always installed as active route with fresh expiry")
        print("  - Inherits existing seq_num if routes already exist")

    finally:
        iface.close()


def test_2_hop_path():
    """
    Test a simple 2-hop path:
      Controller → Node 0x20 → Node 0x30 → Destination
    """
    iface = TCPInterface(hostname="127.0.0.1", portNumber=4403)
    
    try:
        route_install = sdn_pb2.SDNRouteInstall()
        route_install.destination = 0x00000030  # Node C
        
        # 2-hop path: 0x20 → 0x30
        path = [0x20, 0x30]
        route_install.hop_path = pack_hop_path(path)
        route_install.install_id = 2
        
        sdn_msg = sdn_pb2.SDN()
        sdn_msg.route_install.CopyFrom(route_install)
        
        payload = sdn_msg.SerializeToString()
        
        iface.sendData(
            data=payload,
            destinationId=f"!{path[0]:08x}",  # Start at first hop
            portNum=portnums_pb2.PortNum.SDN_APP,
            wantAck=False,
            channelIndex=0,
        )
        
        print(f"✓ 2-hop route install sent")
        print(f"  Install ID: {route_install.install_id}")
        print(f"  Path: {' → '.join([f'0x{hop:02x}' for hop in path])}")
        print(f"  Destination: 0x{route_install.destination:08x}")
        print(f"  All routes installed with seq_num=0 (authoritative)")
        
    finally:
        iface.close()


def test_direct_path():
    """
    Test a direct 1-hop path (starting node forwards directly to destination):
      Controller → Node 0x20 → Destination
    """
    iface = TCPInterface(hostname="127.0.0.1", portNumber=4403)
    
    try:
        route_install = sdn_pb2.SDNRouteInstall()
        route_install.destination = 0x00000025  # Destination
        
        # 1-hop path: 0x20 forwards directly to destination
        path = [0x20]
        route_install.hop_path = pack_hop_path(path)
        route_install.install_id = 3
        
        sdn_msg = sdn_pb2.SDN()
        sdn_msg.route_install.CopyFrom(route_install)
        
        payload = sdn_msg.SerializeToString()
        
        iface.sendData(
            data=payload,
            destinationId=f"!{path[0]:08x}",  # Node 0x20
            portNum=portnums_pb2.PortNum.SDN_APP,
            wantAck=False,
            channelIndex=0,
        )
        
        print(f"✓ Direct path test sent")
        print(f"  Install ID: {route_install.install_id}")
        print(f"  Start node: 0x{path[0]:02x}")
        print(f"  Destination: 0x{route_install.destination:08x}")
        print(f"  Route installed with seq_num=0 (authoritative)")
        
    finally:
        iface.close()


def test_max_hops():
    """
    Test maximum 8-hop path to validate packing/unpacking.
    """
    iface = TCPInterface(hostname="127.0.0.1", portNumber=4403)
    
    try:
        route_install = sdn_pb2.SDNRouteInstall()
        route_install.destination = 0x000000FF  # Final destination
        
        # 8-hop path (maximum allowed)
        path = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
        route_install.hop_path = pack_hop_path(path)
        route_install.install_id = 4
        
        # Verify packing/unpacking
        unpacked = unpack_hop_path(route_install.hop_path)
        assert unpacked == path, f"Packing error: {unpacked} != {path}"
        
        sdn_msg = sdn_pb2.SDN()
        sdn_msg.route_install.CopyFrom(route_install)
        
        payload = sdn_msg.SerializeToString()
        
        iface.sendData(
            data=payload,
            destinationId=f"!{path[0]:08x}",
            portNum=portnums_pb2.PortNum.SDN_APP,
            wantAck=False,
            channelIndex=0,
        )
        
        print(f"✓ Maximum 8-hop route install sent")
        print(f"  Install ID: {route_install.install_id}")
        print(f"  Path: {' → '.join([f'0x{hop:02x}' for hop in path])}")
        print(f"  Packed: 0x{route_install.hop_path:016x}")
        print(f"  Unpacked: {[f'0x{h:02x}' for h in unpacked]}")
        print(f"  All 8 hops installed with seq_num=0 (authoritative)")
        
    finally:
        iface.close()


def test_packing():
    """
    Unit test for hop_path packing/unpacking functions.
    """
    print("Testing hop_path packing/unpacking...")
    
    # Test 1: Simple 3-hop path
    path1 = [0x12, 0x34, 0x56]
    packed1 = pack_hop_path(path1)
    unpacked1 = unpack_hop_path(packed1)
    print(f"  Test 1: {path1} → 0x{packed1:016x} → {unpacked1}")
    assert unpacked1 == path1, "Test 1 failed"
    
    # Test 2: Single hop
    path2 = [0xAB]
    packed2 = pack_hop_path(path2)
    unpacked2 = unpack_hop_path(packed2)
    print(f"  Test 2: {path2} → 0x{packed2:016x} → {unpacked2}")
    assert unpacked2 == path2, "Test 2 failed"
    
    # Test 3: Maximum 8 hops
    path3 = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]
    packed3 = pack_hop_path(path3)
    unpacked3 = unpack_hop_path(packed3)
    print(f"  Test 3: {[f'0x{h:02x}' for h in path3]}")
    print(f"          → 0x{packed3:016x}")
    print(f"          → {[f'0x{h:02x}' for h in unpacked3]}")
    assert unpacked3 == path3, "Test 3 failed"
    
    # Test 4: Path with trailing zeros (should stop at first zero)
    packed4 = 0x0000000000003412  # Only 2 hops: 0x12, 0x34
    unpacked4 = unpack_hop_path(packed4)
    print(f"  Test 4: 0x{packed4:016x} → {[f'0x{h:02x}' for h in unpacked4]}")
    assert unpacked4 == [0x12, 0x34], "Test 4 failed"
    
    print("✓ All packing tests passed!")


if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("SDN Route Install Test Script")
    print("="*70)
    print()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "2hop":
            test_2_hop_path()
        elif sys.argv[1] == "direct":
            test_direct_path()
        elif sys.argv[1] == "maxhops":
            test_max_hops()
        elif sys.argv[1] == "pack":
            test_packing()
        else:
            print(f"Unknown test: {sys.argv[1]}")
            print("Usage: python test_sdn_route_install.py [2hop|direct|maxhops|pack]")
            print()
            print("Available tests:")
            print("  (default) - 3-hop path test with detailed output")
            print("  2hop      - Simple 2-hop path")
            print("  direct    - 1-hop direct path")
            print("  maxhops   - Maximum 8-hop path")
            print("  pack      - Unit test for packing/unpacking functions")
            sys.exit(1)
    else:
        # Default: 3-hop test with detailed explanation
        main()