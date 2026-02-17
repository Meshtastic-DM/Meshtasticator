#!/bin/bash
# Helper script to run the Meshtastic Interactive Simulator

cd "$(dirname "$0")"

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found. Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Clean up old node processes
pkill -f "portduino/node" 2>/dev/null
rm -f ~/node*.log 2>/dev/null

# Run the simulator
echo "Starting Interactive Simulator..."
echo "Nodes will open in xterm windows (or run headless if xterm unavailable)"
echo ""

python3 interactiveSim.py -p ../../Firmware_meshtastic/firmware_meshtastic_new/.pio/build/native/

# Clean up on exit
echo ""
echo "Cleaning up..."
pkill -f "portduino/node" 2>/dev/null
