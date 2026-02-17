#!/bin/bash
# filepath: /home/raveen/Final_Year_Project/Simulator_new/Meshtasticator/generate_python_protos.sh
# Script to generate Python protobuf files from .proto definitions

set -e

echo "Generating Python protobuf files for AODV and Meshtastic..."

PROTO_DIR="/home/raveen/firmware_meshtastic_new/protobufs"
OUTPUT_DIR="/home/raveen/Final_Year_Project/Simulator_new/Meshtasticator"

cd "$PROTO_DIR"

# Check if we have protoc available
if command -v protoc &> /dev/null; then
    echo "Using system protoc..."
    
    # Generate all necessary protobufs
    protoc --python_out="$OUTPUT_DIR" -I. meshtastic/aodv.proto
    protoc --python_out="$OUTPUT_DIR" -I. meshtastic/mesh.proto
    protoc --python_out="$OUTPUT_DIR" -I. meshtastic/portnums.proto
    
elif command -v python3 &> /dev/null && python3 -c "import grpc_tools.protoc" 2>/dev/null; then
    echo "Using grpc_tools.protoc..."
    
    python3 -m grpc_tools.protoc -I. --python_out="$OUTPUT_DIR" meshtastic/aodv.proto
    python3 -m grpc_tools.protoc -I. --python_out="$OUTPUT_DIR" meshtastic/mesh.proto
    python3 -m grpc_tools.protoc -I. --python_out="$OUTPUT_DIR" meshtastic/portnums.proto
else
    echo "Error: Neither 'protoc' nor 'grpc_tools.protoc' found."
    echo ""
    echo "Please install one of the following:"
    echo "  Option 1: Install protobuf compiler:"
    echo "    sudo apt install protobuf-compiler"
    echo ""
    echo "  Option 2: Install Python grpcio-tools:"
    echo "    pip install grpcio-tools"
    exit 1
fi

# Check if files were created
if [ -f "$OUTPUT_DIR/meshtastic/aodv_pb2.py" ]; then
    echo "✓ Successfully generated protobuf files"
else
    echo "✗ Failed to generate protobuf files"
    exit 1
fi

echo ""
echo "Generated files are in: $OUTPUT_DIR/meshtastic/"