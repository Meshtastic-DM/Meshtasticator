#!/usr/bin/env python3
"""
Meshtastic TCP Packet Monitor
Connects to a Meshtastic node via TCP and displays received packets.
Includes SDN and AODV packet support.
"""

import meshtastic
import meshtastic.tcp_interface
from pubsub import pub
from datetime import datetime

# TCP connection details
HOST = "127.0.0.1"
PORT = 4408

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
        print(f"Packet ID: {packet['id']}")
    
    # Decoded data
    if 'decoded' in packet:
        decoded = packet['decoded']
        portnum = decoded.get('portnum', 'Unknown')
        print(f"Port Number: {portnum}")
        
        # Text messages
        if portnum == 'TEXT_MESSAGE_APP':
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
        
        # SDN packets
        elif portnum == 'SDN_APP':
            print(f"🌐 SDN PACKET:")
            if 'payload' in decoded:
                # Try to parse as SDN protobuf
                try:
                    # The payload should contain the SDN message
                    # Check for different SDN message types by looking at decoded data
                    if 'announcement' in str(decoded):
                        print(f"  Type: SDN Announcement")
                    elif 'routeUpdate' in str(decoded):
                        print(f"  Type: Route Update")
                    elif 'routeCommand' in str(decoded):
                        print(f"  Type: Route Command")
                    elif 'routeInstall' in str(decoded):
                        print(f"  Type: Route Install")
                    elif 'linkQuality' in str(decoded):
                        print(f"  Type: Link Quality Report")
                        # Try to extract link quality data if available
                        data = decoded.get('payload', {})
                        if hasattr(data, 'relayNode'):
                            print(f"  Relay Nodes: {[hex(n) for n in data.relayNode]}")
                            print(f"  RX Good: {list(data.rxGood)}")
                            print(f"  RX Bad: {list(data.rxBad)}")
                            if hasattr(data, 'channelUtilization'):
                                print(f"  Channel Util: {data.channelUtilization:.1f}%")
                            if hasattr(data, 'airUtilTx'):
                                print(f"  Air Util TX: {data.airUtilTx:.1f}%")
                    else:
                        print(f"  Type: Unknown SDN message")
                        print(f"  Payload: {decoded.get('payload', 'N/A')}")
                except Exception as e:
                    print(f"  Raw payload length: {len(decoded.get('payload', []))} bytes")
                    print(f"  (Parse error: {e})")
            else:
                print(f"  (No payload data)")
        
        # AODV packets
        elif portnum == 'AODV_APP':
            print(f"🗺️  AODV PACKET:")
            if 'payload' in decoded:
                try:
                    # Try to determine AODV message type from payload
                    payload = decoded.get('payload', b'')
                    if isinstance(payload, bytes) and len(payload) > 0:
                        # First byte often indicates message type in AODV
                        msg_type = payload[0] if len(payload) > 0 else 0
                        if msg_type == 1:
                            print(f"  Type: Route Request (RREQ)")
                        elif msg_type == 2:
                            print(f"  Type: Route Reply (RREP)")
                        elif msg_type == 3:
                            print(f"  Type: Route Error (RERR)")
                        else:
                            print(f"  Type: Unknown AODV message (type={msg_type})")
                        print(f"  Payload length: {len(payload)} bytes")
                    else:
                        print(f"  (Empty or non-bytes payload)")
                except Exception as e:
                    print(f"  Raw payload: {decoded.get('payload', 'N/A')}")
                    print(f"  (Parse error: {e})")
            else:
                print(f"  (No payload data)")
        
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
    print(f"✅ Connected to {HOST}:{PORT}")
    print("Listening for packets (including SDN & AODV)...\n")

def on_disconnect(interface, topic=pub.AUTO_TOPIC):
    """Callback when disconnected"""
    print(f"\n❌ Disconnected from {HOST}:{PORT}")

def main():
    print(f"Connecting to Meshtastic node at {HOST}:{PORT}...")
    
    try:
        # Subscribe to packet events BEFORE creating interface
        pub.subscribe(on_receive, "meshtastic.receive")
        pub.subscribe(on_connection, "meshtastic.connection.established")
        pub.subscribe(on_disconnect, "meshtastic.connection.lost")
        
        # Create TCP interface
        interface = meshtastic.tcp_interface.TCPInterface(hostname=HOST, portNumber=PORT)
        
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
