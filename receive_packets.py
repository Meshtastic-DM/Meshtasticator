#!/usr/bin/env python3
"""
Meshtastic TCP Packet Monitor
Connects to a Meshtastic node via TCP and displays received packets.
"""

import meshtastic
import meshtastic.tcp_interface
from pubsub import pub
from datetime import datetime

# TCP connection details
HOST = "127.0.0.1"
PORT = 4403

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
        print(f"Port Number: {decoded.get('portnum', 'Unknown')}")
        
        # Text messages
        if decoded.get('portnum') == 'TEXT_MESSAGE_APP':
            text = decoded.get('text', decoded.get('payload', b'')).decode('utf-8', errors='replace') if isinstance(decoded.get('text', decoded.get('payload', '')), bytes) else decoded.get('text', '')
            print(f"📨 MESSAGE: {text}")
        
        # Position data
        elif decoded.get('portnum') == 'POSITION_APP' and 'position' in decoded:
            pos = decoded['position']
            lat = pos.get('latitude', pos.get('latitudeI', 0)) / 1e7 if 'latitudeI' in pos else pos.get('latitude', 0)
            lon = pos.get('longitude', pos.get('longitudeI', 0)) / 1e7 if 'longitudeI' in pos else pos.get('longitude', 0)
            alt = pos.get('altitude', 0)
            print(f"📍 POSITION: Lat={lat:.6f}, Lon={lon:.6f}, Alt={alt}m")
        
        # Telemetry
        elif decoded.get('portnum') == 'TELEMETRY_APP' and 'telemetry' in decoded:
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
        elif decoded.get('portnum') == 'NODEINFO_APP' and 'user' in decoded:
            user = decoded['user']
            print(f"👤 NODE INFO:")
            print(f"  Long Name: {user.get('longName', 'Unknown')}")
            print(f"  Short Name: {user.get('shortName', 'Unknown')}")
            print(f"  Hardware: {user.get('hwModel', 'Unknown')}")
        
        else:
            print(f"Other packet type: {decoded.get('portnum')}")
    
    # Signal quality
    if 'rxSnr' in packet:
        print(f"SNR: {packet['rxSnr']} dB")
    if 'rxRssi' in packet:
        print(f"RSSI: {packet['rxRssi']} dBm")
    if 'hopLimit' in packet:
        print(f"Hop Limit: {packet['hopLimit']}")

def on_connection(interface, topic=pub.AUTO_TOPIC):
    """Callback when connection is established"""
    print(f"✅ Connected to {HOST}:{PORT}")
    print("Listening for packets...\n")

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