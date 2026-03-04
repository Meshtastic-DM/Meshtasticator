#!/usr/bin/env python3
"""
Meshtastic Serial Packet Monitor
Connects to a Meshtastic node via serial COM port and displays received packets.
Includes SDN and AODV packet support.
"""

import meshtastic
import meshtastic.serial_interface
from pubsub import pub
from datetime import datetime
# Protobuf imports
from generated import sdn_pb2, aodv_pb2, portnums_pb2

# Serial connection details
COM_PORT = "COM8"
USE_SERIAL = True

def on_receive(packet, interface):
    """Callback for all received packets"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Packet Received")
    print(f"{'='*60}")

    # Basic packet info
    if 'from' in packet:
        print(f"From Node: {packet['from']} (0x{packet['from']:08x})")
    if 'to' in packet:
        print(f"To Node: {packet['to']} (0x{packet['to']:08x})")
    if 'id' in packet:
        print(f"Packet ID: {packet['id']} (0x{packet['id']:08x})")

    # Decoded data
    if 'decoded' in packet:
        decoded = packet['decoded']
        portnum = decoded.get('portnum', 'Unknown')
        print(f"Port Number: {portnum}")

        # Use portnums_pb2 for robust port number matching
        try:
            portnum_val = int(portnum) if isinstance(portnum, int) else getattr(portnums_pb2.PortNum, str(portnum), None)
        except Exception:
            portnum_val = None

        # SDN packets
        if portnum_val == portnums_pb2.PortNum.SDN_APP:
            print(f"🌐 SDN PACKET:")
            payload = decoded.get('payload', b'')
            try:
                sdn_msg = sdn_pb2.SDN()
                sdn_msg.ParseFromString(payload)
                # Display SDN message type
                if sdn_msg.HasField("announcement"):
                    print("  Type: SDN Announcement")
                    print(f"  Announcement: {sdn_msg.announcement}")
                elif sdn_msg.HasField("route_update"):
                    print("  Type: Route Update")
                    print(f"  RouteUpdate: {sdn_msg.route_update}")
                elif sdn_msg.HasField("route_command"):
                    print("  Type: Route Command")
                    print(f"  RouteCommand: {sdn_msg.route_command}")
                elif sdn_msg.HasField("route_install"):
                    print("  Type: Route Install")
                    print(f"  RouteInstall: {sdn_msg.route_install}")
                elif sdn_msg.HasField("route_set"):
                    print("  Type: Route Set")
                    print(f"  RouteSet: {sdn_msg.route_set}")
                elif sdn_msg.HasField("route_set_confirm"):
                    print("  Type: Route Set Confirm")
                    print(f"  RouteSetConfirm: {sdn_msg.route_set_confirm}")
                elif sdn_msg.HasField("link_quality"):
                    print("  Type: Link Quality Report")
                    lq = sdn_msg.link_quality
                    print(f"  Relay Nodes: {[hex(n) for n in lq.relay_node]}")
                    print(f"  RX Good: {list(lq.rx_good)}")
                    print(f"  RX Bad: {list(lq.rx_bad)}")
                    print(f"  Channel Util: {getattr(lq, 'channel_utilization', 'N/A')}")
                    print(f"  Air Util TX: {getattr(lq, 'air_util_tx', 'N/A')}")
                else:
                    print("  Type: Unknown SDN message")
                    print(f"  SDN Raw: {sdn_msg}")
            except Exception as e:
                print(f"  (SDN Parse error: {e})")
                print(f"  Raw payload length: {len(payload)} bytes")

        # AODV packets
        elif portnum_val == portnums_pb2.PortNum.AODV_ROUTING_APP:
            print(f"🗺️  AODV PACKET:")
            payload = decoded.get('payload', b'')
            try:
                aodv_msg = aodv_pb2.AODV()
                aodv_msg.ParseFromString(payload)
                if aodv_msg.HasField("rreq"):
                    print("  Type: Route Request (RREQ)")
                    print(f"  RREQ: {aodv_msg.rreq}")
                elif aodv_msg.HasField("rrep"):
                    print("  Type: Route Reply (RREP)")
                    print(f"  RREP: {aodv_msg.rrep}")
                elif aodv_msg.HasField("rerr"):
                    print("  Type: Route Error (RERR)")
                    print(f"  RERR: {aodv_msg.rerr}")
                else:
                    print("  Type: Unknown AODV message")
                    print(f"  AODV Raw: {aodv_msg}")
            except Exception as e:
                print(f"  (AODV Parse error: {e})")
                print(f"  Raw payload length: {len(payload)} bytes")

        # Text messages
        elif portnum == 'TEXT_MESSAGE_APP':
            text = decoded.get('text', decoded.get('payload', b'')).decode('utf-8', errors='replace') if isinstance(decoded.get('text', decoded.get('payload', '')), bytes) else decoded.get('text', '')
            print(f"📨 MESSAGE: {text}")

        # Position data
        elif portnum == 'POSITION_APP' and 'position' in decoded:
            pos = decoded['position']
            lat = pos.get('latitude', pos.get('latitudeI', 0)) / 1e7 if 'latitudeI' in pos else pos.get('latitude', 0)
            lon = pos.get('longitude', pos.get('longitudeI', 0)) / 1e7 if 'longitudeI' in pos else pos.get('longitude', 0)
            alt = pos.get('altitude', 0)
            print(f"📍 POSITION: Lat={lat:.6f}, Lon={lon:.6f}, Alt={alt}m")

        # Telemetry
        elif portnum == 'TELEMETRY_APP' and 'telemetry' in decoded:
            telem = decoded['telemetry']
            if 'deviceMetrics' in telem:
                metrics = telem['deviceMetrics']
                print(f"📊 TELEMETRY:")
                if 'batteryLevel' in metrics:
                    print(f"  Battery: {metrics['batteryLevel']}%")
                if 'voltage' in metrics:
                    print(f"  Voltage: {metrics['voltage']:.2f}V")
                if 'channelUtilization' in metrics:
                    print(f"  Channel Util: {metrics['channelUtilization']:.1f}%")
                if 'airUtilTx' in metrics:
                    print(f"  Air Util TX: {metrics['airUtilTx']:.1f}%")

        # Node info
        elif portnum == 'NODEINFO_APP' and 'user' in decoded:
            user = decoded['user']
            print(f"👤 NODE INFO:")
            print(f"  Long Name: {user.get('longName', 'Unknown')}")
            print(f"  Short Name: {user.get('shortName', 'Unknown')}")
            print(f"  Hardware: {user.get('hwModel', 'Unknown')}")

        else:
            print(f"Other packet type: {portnum}")

    # Signal quality
    if 'rxSnr' in packet:
        print(f"SNR: {packet['rxSnr']} dB")
    if 'rxRssi' in packet:
        print(f"RSSI: {packet['rxRssi']} dBm")
    if 'hopLimit' in packet:
        print(f"Hop Limit: {packet['hopLimit']}")
    if 'hopStart' in packet:
        print(f"Hop Start: {packet['hopStart']}")
    if 'relayNode' in packet:
        print(f"Relay Node: 0x{packet['relayNode']:02x}")

def on_connection(interface, topic=pub.AUTO_TOPIC):
    """Callback when connection is established"""
    print(f"✅ Connected to {COM_PORT}")
    print("Listening for packets (including SDN & AODV)...\n")

def on_disconnect(interface, topic=pub.AUTO_TOPIC):
    """Callback when disconnected"""
    print(f"\n❌ Disconnected from {COM_PORT}")

def main():
    print(f"Connecting to Meshtastic node at {COM_PORT}...")
    
    try:
        # Subscribe to packet events BEFORE creating interface
        pub.subscribe(on_receive, "meshtastic.receive")
        pub.subscribe(on_connection, "meshtastic.connection.established")
        pub.subscribe(on_disconnect, "meshtastic.connection.lost")
        
        # Create serial interface
        interface = meshtastic.serial_interface.SerialInterface(devPath=COM_PORT)
        
        print("\nPress Ctrl+C to exit\n")
        print("📡 Monitoring for:")
        print("  • Text Messages")
        print("  • Position Updates")
        print("  • Telemetry")
        print("  • Node Info")
        print("  • SDN Packets (Announcements, Routes, Link Quality)")
        print("  • AODV Packets (RREQ, RREP, RERR)")
        print("")
        
        # Keep the script running
        while True:
            pass
            
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        interface.close()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
