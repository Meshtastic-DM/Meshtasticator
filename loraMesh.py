#!/usr/bin/env python3
import csv
import os
import sys
import random

from matplotlib import pyplot as plt
import yaml
import simpy
import numpy as np

from lib import phy
from lib.common import Graph, plot_schedule, gen_scenario, run_graph_updates, setup_asymmetric_links
from lib.config import Config
from lib.discrete_event import BroadcastPipe
from lib.node import MeshNode
from lib.node_aodv import MeshNode_AODV
from lib.node_zrp import MeshNode_ZRP

VERBOSE = False
conf = Config()
random.seed(conf.SEED)


def verboseprint(*args, **kwargs):
	if VERBOSE:
		print(*args, **kwargs)


def parse_params(conf, args):
	# TODO: refactor with argparse
	routing_type = None
	
	# Check for --route-type argument
	if "--route-type" in args:
		route_type_index = args.index("--route-type")
		if route_type_index + 1 < len(args):
			routing_type = args[route_type_index + 1]
			# Remove --route-type and its value from args
			args = args[:route_type_index] + args[route_type_index + 2:]
		else:
			print("Error: --route-type requires a value")
			exit(1)
	
	if len(args) > 3:
		print("Usage: ./loraMesh [nr_nodes] [--from-file [file_name]] [--route-type ROUTING_TYPE]")
		print("Do not specify the number of nodes when reading from a file.")
		exit(1)
	else:
		if len(args) > 1:
			if isinstance(args[1], str) and ("--from-file" in args[1]):
				if len(args) > 2:
					string = args[2]
				else:
					string = 'nodeConfig.yaml'
				
				# Check if path is absolute or already contains directory separators
				if os.path.isabs(string) or os.path.dirname(string):
					config_path = string
				else:
					config_path = os.path.join("out", string)
				
				with open(config_path, 'r') as file:
					config = yaml.load(file, Loader=yaml.FullLoader)
				
				# Apply routing type if specified
				if routing_type:
					try:
						routerType = conf.ROUTER_TYPE(routing_type)
						conf.SELECTED_ROUTER_TYPE = routerType
						conf.update_router_dependencies()
					except ValueError:
						valid_types = [member.name for member in conf.ROUTER_TYPE]
						print(f"Invalid router type: {routing_type}")
						print(f"Router type must be one of: {', '.join(valid_types)}")
						exit(1)
			else:
				conf.NR_NODES = int(args[1])
				config = [None for _ in range(conf.NR_NODES)]
				if len(args) > 2:
					try:
						# Attempt to convert the string args[2] into a valid enum member
						routerType = conf.ROUTER_TYPE(args[2])
						conf.SELECTED_ROUTER_TYPE = routerType
						conf.update_router_dependencies()
					except ValueError:
						# If it fails, print possible values
						valid_types = [member.name for member in conf.ROUTER_TYPE]
						print(f"Invalid router type: {args[2]}")
						print(f"Router type must be one of: {', '.join(valid_types)}")
						exit(1)
				elif routing_type:
					# Use --route-type if no positional routing type provided
					try:
						routerType = conf.ROUTER_TYPE(routing_type)
						conf.SELECTED_ROUTER_TYPE = routerType
						conf.update_router_dependencies()
					except ValueError:
						valid_types = [member.name for member in conf.ROUTER_TYPE]
						print(f"Invalid router type: {routing_type}")
						print(f"Router type must be one of: {', '.join(valid_types)}")
						exit(1)
				if conf.NR_NODES == -1:
					config = gen_scenario(conf)
		else:
			config = gen_scenario(conf)
		if config[0] is not None:
			conf.NR_NODES = len(config.keys())
		if conf.NR_NODES < 2:
			print("Need at least two nodes.")
			exit(1)

	print("Number of nodes:", conf.NR_NODES)
	print("Modem:", conf.MODEM)
	print("Simulation time (s):", conf.SIMTIME/1000)
	print("Period (s):", conf.PERIOD/1000)
	print("Interference level:", conf.INTERFERENCE_LEVEL)
	return config


nodeConfig = parse_params(conf, sys.argv)
conf.update_router_dependencies()
env = simpy.Environment()
bc_pipe = BroadcastPipe(env)

# simulation variables
nodes = []
messages = []
packets = []
delays = []
packetsAtN = [[] for _ in range(conf.NR_NODES)]
messageSeq = {"val": 0}
totalPairs = 0
symmetricLinks = 0
asymmetricLinks = 0
noLinks = 0

graph = Graph(conf)
if conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.AODV:
	for i in range(conf.NR_NODES):
		node = MeshNode_AODV(conf, nodes, env, bc_pipe, i, conf.PERIOD, messages, packetsAtN, packets, delays, nodeConfig[i], messageSeq, verboseprint)
		nodes.append(node)
		graph.add_node(node)
elif conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.SDN_AODV:
	from lib.node_sdn import MeshNode_SDN
	for i in range(conf.NR_NODES):
		node = MeshNode_SDN(conf, nodes, env, bc_pipe, i, conf.PERIOD, messages, packetsAtN, packets, delays, nodeConfig[i], messageSeq, verboseprint)
		nodes.append(node)
		graph.add_node(node)
elif conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.ZRP:
	for i in range(conf.NR_NODES):
		node = MeshNode_ZRP(conf, nodes, env, bc_pipe, i, conf.PERIOD, messages, packetsAtN, packets, delays, nodeConfig[i], messageSeq, verboseprint)
		nodes.append(node)
		graph.add_node(node)
else:
	for i in range(conf.NR_NODES):
		node = MeshNode(conf, nodes, env, bc_pipe, i, conf.PERIOD, messages, packetsAtN, packets, delays, nodeConfig[i], messageSeq, verboseprint)
		nodes.append(node)
		graph.add_node(node)

totalPairs, symmetricLinks, asymmetricLinks, noLinks = setup_asymmetric_links(conf, nodes)

if conf.MOVEMENT_ENABLED:
	env.process(run_graph_updates(env, graph, nodes, conf.ONE_MIN_INTERVAL))

conf.update_router_dependencies()

# start simulation
print("\n====== START OF SIMULATION ======")
env.run(until=conf.SIMTIME)

# compute statistics
print("\n====== END OF SIMULATION ======")
print("*******************************")
print(f"\nRouter Type: {conf.SELECTED_ROUTER_TYPE}")
print('Number of messages created:', messageSeq["val"])
sent = len(packets)
if conf.DMs:
	potentialReceivers = sent
else:
	potentialReceivers = sent*(conf.NR_NODES-1)
print('Number of packets sent:', sent, 'to', potentialReceivers, 'potential receivers')
nrCollisions = sum([1 for p in packets for n in nodes if p.collidedAtN[n.nodeid] is True])
print("Number of collisions:", nrCollisions)
nrSensed = sum([1 for p in packets for n in nodes if p.sensedByN[n.nodeid] is True])
print("Number of packets sensed:", nrSensed)
nrReceived = sum([1 for p in packets for n in nodes if p.receivedAtN[n.nodeid] is True])
print("Number of packets received:", nrReceived)
meanDelay = np.nanmean(delays)
print('Delay average (ms):', round(meanDelay, 2))
txAirUtilization = sum([n.txAirUtilization for n in nodes])/conf.NR_NODES/conf.SIMTIME*100
print('Average Tx air utilization:', round(txAirUtilization, 2), '%')
if nrSensed != 0:
	collisionRate = float((nrCollisions)/nrSensed)
	print("Percentage of packets that collided:", round(collisionRate*100, 2))
else:
	print("No packets sensed.")
nodeReach = sum([n.usefulPackets for n in nodes])/(messageSeq["val"]*(conf.NR_NODES-1))
print("Average percentage of nodes reached:", round(nodeReach*100, 2))
if nrReceived != 0:
	usefulness = sum([n.usefulPackets for n in nodes])/nrReceived  # nr of packets that delivered to a packet to a new receiver out of all packets sent
	print("Percentage of received packets containing new message:", round(usefulness*100, 2))
else:
	print('No packets received.')
delayDropped = sum(n.droppedByDelay for n in nodes)
print("Number of packets dropped by delay/hop limit:", delayDropped)

if conf.MODEL_ASYMMETRIC_LINKS:
	print("Asymmetric links:", round(asymmetricLinks / totalPairs * 100, 2), '%')
	print("Symmetric links:", round(symmetricLinks / totalPairs * 100, 2), '%')
	print("No links:", round(noLinks / totalPairs * 100, 2), '%')

if conf.MOVEMENT_ENABLED:
	movingNodes = sum([1 for n in nodes if n.isMoving is True])
	print("Number of moving nodes:", movingNodes)
	gpsEnabled = sum([1 for n in nodes if n.gpsEnabled is True])
	print("Number of moving nodes w/ GPS:", gpsEnabled)

graph.save()

for node in nodes:
    print(node)

    # ---------- AODV routing table ----------
    if isinstance(node, MeshNode_AODV):
        routeTable = node.get_route_table()
        if len(routeTable) > 0:
            print(f"\nNode {node.nodeid} AODV route table:")
            for dest, entry in routeTable.items():
                print(
                    f"  Dest: {dest}, "
                    f"Next Hop: {entry['nextHop']}, "
                    f"Hop Count: {entry['hopCount']}, "
                    f"Seq: {entry['destSeqNum']}"
                )
        else:
            print(f"\nNode {node.nodeid} has an empty AODV route table.")

    # ---------- ZRP IARP + IERP tables ----------
    elif isinstance(node, MeshNode_ZRP):
        # IARP
        iarpTable = node.get_iarp_table()
        if len(iarpTable) > 0:
            print(f"\nNode {node.nodeid} ZRP IARP table:")
            for dest, entry in iarpTable.items():
                print(
                    f"  Dest: {dest}, "
                    f"Next Hop: {entry['nextHop']}, "
                    f"Distance: {entry['distance']}, "
                    f"Seq: {entry['seq_num']}"
                )
        else:
            print(f"\nNode {node.nodeid} has an empty ZRP IARP table.")

        # IERP (coarse inter-zone info)
        ierpTable = node.get_ierp_table()
        if len(ierpTable) > 0:
            print(f"\nNode {node.nodeid} ZRP IERP table:")
            for dest, entry in ierpTable.items():
                print(
                    f"  Dest: {dest}, "
                    f"Next Hop: {entry['nextHop']}, "
                    f"Distance: {entry['distance']}, "
                    f"Seq: {entry['seq_num']}"
                )
        else:
            print(f"Node {node.nodeid} has an empty ZRP IERP table.")


    else:
        print(f"\nNode {node.nodeid} has no routing/IARP table interface.")


if conf.PLOT:
	plot_schedule(conf, packets, messages)

sensorPacketMeanDelays = {}
sensorPacketsDelayArrays = {}
dmPacketMeanDelays = {}
dmPacketsDelayArrays = {}
brocastPacketMeanDelays = {}
brocastPacketsDelayArrays = {}

for n in nodes:
	sensorPacketsDelays = n.SensorPacketsDelays
	for originTxNodeId, delays in sensorPacketsDelays.items():
		if delays:
			if originTxNodeId not in sensorPacketMeanDelays:
				sensorPacketMeanDelays[originTxNodeId] = {}
				sensorPacketsDelayArrays[originTxNodeId] = {}
			meanDelay = np.nanmean(delays)
			sensorPacketMeanDelays[originTxNodeId][n.nodeid] = meanDelay
			sensorPacketsDelayArrays[originTxNodeId][n.nodeid] = delays

			print(f"Average delay of sensor packets from node {originTxNodeId} to node {n.nodeid} (ms):", round(meanDelay, 2))

	broadCastPacketsDelays = n.BroadcastPacketsDelays
	for originTxNodeId, delays in broadCastPacketsDelays.items():
		if delays:
			if originTxNodeId not in brocastPacketMeanDelays:
				brocastPacketMeanDelays[originTxNodeId] = {}
				brocastPacketsDelayArrays[originTxNodeId] = {}
			meanDelay = np.nanmean(delays)
			brocastPacketMeanDelays[originTxNodeId][n.nodeid] = meanDelay
			brocastPacketsDelayArrays[originTxNodeId][n.nodeid] = delays
			print(f"Average delay of broadcast packets from node {originTxNodeId} to node {n.nodeid} (ms):", round(meanDelay, 2))
	dmPacketsDelays = n.DMPacketsDelays
	for originTxNodeId, delays in dmPacketsDelays.items():
		if delays:
			if originTxNodeId not in dmPacketMeanDelays:
				dmPacketMeanDelays[originTxNodeId] = {}
				dmPacketsDelayArrays[originTxNodeId] = {}
			meanDelay = np.nanmean(delays)
			dmPacketMeanDelays[originTxNodeId][n.nodeid] = meanDelay
			dmPacketsDelayArrays[originTxNodeId][n.nodeid] = delays
			print(f"Average delay of DM packets from node {originTxNodeId} to node {n.nodeid} (ms):", round(meanDelay, 2))
	
CreatedDMPackets = {}
RecivedDMPackets = {}
CreatedSensorPackets = {}
RecivedSensorPackets = {}
RecivedBroadcastPackets = {}
BroadcastPacketsExtra = {}
DMPacketsExtra = {}
SensorPacketsExtra = {}
TotalCreatedPackets = 0

for n in nodes:
    if n.simRole == "Control_Center":
        for origId, packet in n.SensorPacketsReceivedOrigId.items():
            if origId not in RecivedSensorPackets:
                RecivedSensorPackets[origId] = {}
                SensorPacketsExtra[origId] = {}
            RecivedSensorPackets[origId][n.nodeid] = len(packet.keys())
            SensorPacketsExtra[origId][n.nodeid] = sum([count - 1 for count in packet.values() if count > 1])  # count extra packets received

    if n.simRole == "DM" or n.simRole == "Control_Center":
        if n.nodeid not in CreatedDMPackets:
            CreatedDMPackets[n.nodeid] = {}
        for destId, count in n.numberOfDMPacketsCreated.items():
            CreatedDMPackets[n.nodeid][destId] = count

        for origId, packet in n.DMPacketsReceivedOrigId.items():
            if origId not in RecivedDMPackets:
                RecivedDMPackets[origId] = {}
                DMPacketsExtra[origId] = {}
            RecivedDMPackets[origId][n.nodeid] = len(packet.keys())
            DMPacketsExtra[origId][n.nodeid] = sum([count - 1 for count in packet.values() if count > 1])  # count extra packets received

    elif n.simRole == "Sensor":
        if n.nodeid not in CreatedSensorPackets:
            CreatedSensorPackets[n.nodeid] = {}
        for destId, count in n.numberOfSensorPacketsCreated.items():
            CreatedSensorPackets[n.nodeid][destId] = count

    # These lines apply to all nodes
    RecivedBroadcastPackets[n.nodeid] = len(n.BroadcastPacketsReceived.keys())
    BroadcastPacketsExtra[n.nodeid] = sum([count - 1 for count in n.BroadcastPacketsReceived.values() if count > 1])  # count extra packets received
    TotalCreatedPackets += n.numberOfBroadcastPacketsCreated

def sort_nested_dictionary(nested_dict):
    """Sort both outer and inner dictionaries by keys"""
    return {
        outer_key: dict(sorted(inner_dict.items()))
        for outer_key, inner_dict in sorted(nested_dict.items())
    }
CreatedDMPackets = sort_nested_dictionary(CreatedDMPackets)
RecivedDMPackets = sort_nested_dictionary(RecivedDMPackets)
CreatedSensorPackets = sort_nested_dictionary(CreatedSensorPackets)
RecivedSensorPackets = sort_nested_dictionary(RecivedSensorPackets)
RecivedBroadcastPackets = dict(sorted(RecivedBroadcastPackets.items()))
BroadcastPacketsExtra = dict(sorted(BroadcastPacketsExtra.items()))
DMPacketsExtra = sort_nested_dictionary(DMPacketsExtra)
SensorPacketsExtra = sort_nested_dictionary(SensorPacketsExtra)
print("Number of DM packets created by each node:", CreatedDMPackets)
print("Number of DM packets received by each node:", RecivedDMPackets)
print("Number of sensor packets created by each node:", CreatedSensorPackets)
print("Number of sensor packets received by each node:", RecivedSensorPackets)
print("Total number of broadcast packets created by all nodes:", TotalCreatedPackets)
print("Number of broadcast packets received by each node:", RecivedBroadcastPackets)

print("Number of extra broadcast packets received by each node:", BroadcastPacketsExtra)
print("Number of extra DM packets received by each node:", DMPacketsExtra)
print("Number of extra sensor packets received by each node:", SensorPacketsExtra)

print("Sensor packets delays:", sensorPacketMeanDelays)
print("DM packets delays:", dmPacketMeanDelays)
print("Broadcast packets delays:", brocastPacketMeanDelays)

print("Range of nodes is",phy.MAXRANGE, "m")


N = len(nodes)
realiabilitySensor = [0 for _ in range(N)]
dest = 0
for source in range(N):
	if source != dest:
		if source in CreatedSensorPackets.keys(): 
			if dest in CreatedSensorPackets[source].keys():
				if source in RecivedSensorPackets.keys():
					if dest in RecivedSensorPackets[source].keys():
						realiabilitySensor[source] = RecivedSensorPackets[source][dest] / CreatedSensorPackets[source][dest]
				else:
					realiabilitySensor[source] = 0
			else:
				realiabilitySensor[source] = None
		else:
			realiabilitySensor[source] = None


realibilityMatrix = np.array([val if val is not None else np.nan for val in realiabilitySensor], dtype=float)

print("Reliability of sensor packets from each node to node 0:", realibilityMatrix)



# x = list(range(len(realibilityMatrix)))  # [0, 1, 2, 3, 4]

# plt.figure(figsize=(8, 6))
# bars = plt.bar(x, realibilityMatrix, color='skyblue', edgecolor='black')

# # Add value labels on top of each bar
# for i, val in enumerate(realibilityMatrix):
#     plt.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

# plt.xlabel("Source Node ID")
# plt.ylabel("Reliability to Destination 0")
# plt.title("Reliability from Sensors to Destination Node 0")
# plt.xticks(x, [f"Src {i}" for i in x])
# plt.grid(axis='y')
# plt.tight_layout()
# plt.savefig(f"output/sensor_reliability_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
# import pickle
# with open(f"output/sensor_reliability_{conf.SELECTED_ROUTER_TYPE}.pkl", 'wb') as f:
#     pickle.dump(plt.gcf(), f)
# plt.close()

# Plot reliability only for actual sensor nodes
sensor_ids = sorted(CreatedSensorPackets.keys())
sensor_reliabilities = [realibilityMatrix[node_id] for node_id in sensor_ids]

plt.figure(figsize=(8, 6))

bars = plt.bar(sensor_ids, sensor_reliabilities)

# Add reliability values above bars
for node_id, value in zip(sensor_ids, sensor_reliabilities):
    if not np.isnan(value):
        plt.text(
            node_id,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

plt.xlabel("Sensor Node ID")
plt.ylabel("Reliability to Destination Node 0")
plt.title("Sensor Packet Delivery Reliability to Node 0")

plt.xticks(sensor_ids)
plt.ylim(0, 1.1)

plt.grid(axis="y")
plt.tight_layout()

plt.savefig(
    f"output/sensor_reliability_{conf.SELECTED_ROUTER_TYPE}.png",
    dpi=200,
    bbox_inches="tight"
)

import pickle

with open(
    f"output/sensor_reliability_{conf.SELECTED_ROUTER_TYPE}.pkl",
    "wb"
) as f:
    pickle.dump(plt.gcf(), f)

plt.close()

realiabilityDm = [[0 for _ in range(N)] for _ in range(N)]
for source in range(N):
	for dest in range(N):
		if source != dest:
			if source in DMPacketsExtra.keys():
				if dest in DMPacketsExtra[source].keys():
					realiabilityDm[source][dest] = RecivedDMPackets[source][dest] / CreatedDMPackets[source][dest]
				else:
					realiabilityDm[source][dest] = None
			else:
				realiabilityDm[source][dest] = None

DMmatrix = np.array([[val if val is not None else np.nan for val in row] for row in realiabilityDm], dtype=float)
np.fill_diagonal(DMmatrix, np.nan)

# Plot with annotations
plt.figure(figsize=(8, 6))
plt.imshow(DMmatrix, cmap='YlGnBu', interpolation='nearest')
plt.colorbar(label='Reliability')

for i in range(N):
    for j in range(N):
        if not np.isnan(DMmatrix[i][j]):
            plt.text(j, i, f"{DMmatrix[i][j]:.2f}", ha='center', va='center', color='black')

plt.title("DM Packet Delivery Reliability Matrix")
plt.xlabel("Destination Node ID")
plt.ylabel("Source Node ID")
plt.xticks(ticks=np.arange(N), labels=np.arange(N))
plt.yticks(ticks=np.arange(N), labels=np.arange(N))
plt.grid(False)
plt.tight_layout()
plt.savefig(f"output/dm_reliability_matrix_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
import pickle
with open(f"output/dm_reliability_matrix_{conf.SELECTED_ROUTER_TYPE}.pkl", 'wb') as f:
    pickle.dump(plt.gcf(), f)
plt.close()

# source = None

# for node in nodes:
# 	if node.simRole == "Control_Center":
# 		source = node.nodeid
# 		realibilityBroadcast = [0 for _ in range(N)]
# 		for dest in range(N):
# 			if source != dest:
# 				realibilityBroadcast[dest] = RecivedBroadcastPackets[dest] / TotalCreatedPackets if dest in RecivedBroadcastPackets else None
# 			else:
# 				realibilityBroadcast[dest] = None

# 		realibilityBroadcast = np.array([val if val is not None else np.nan for val in realibilityBroadcast], dtype=float)
# 		break

source = None

for node in nodes:
    if node.simRole == "Control_Center":
        source = node.nodeid
        break

realibilityBroadcast = np.full(N, np.nan, dtype=float)

if source is not None and TotalCreatedPackets > 0:

    realibilityBroadcast = [None for _ in range(N)]

    for dest in range(N):

        if source == dest:
            continue

        if dest in RecivedBroadcastPackets:
            realibilityBroadcast[dest] = (
                RecivedBroadcastPackets[dest] / TotalCreatedPackets
            )

    realibilityBroadcast = np.array(
        [
            val if val is not None else np.nan
            for val in realibilityBroadcast
        ],
        dtype=float
    )

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        range(len(realibilityBroadcast)),
        realibilityBroadcast
    )

    for i, val in enumerate(realibilityBroadcast):
        if not np.isnan(val):
            plt.text(
                i,
                val + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=10
            )

    plt.xlabel("Destination Node ID")
    plt.ylabel("Reliability of Broadcast Packets")
    plt.title("Broadcast Packet Delivery Reliability")

    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(
        f"output/broadcast_reliability_{conf.SELECTED_ROUTER_TYPE}.png",
        dpi=200,
        bbox_inches="tight"
    )

    import pickle

    with open(
        f"output/broadcast_reliability_{conf.SELECTED_ROUTER_TYPE}.pkl",
        "wb"
    ) as f:
        pickle.dump(plt.gcf(), f)

    plt.close()

else:
    print(
        "No broadcast packets were generated; "
        "skipping broadcast reliability calculation."
    )


# if source is not None:
# 	plt.figure(figsize=(8, 6))
# 	bars = plt.bar(range(len(realibilityBroadcast)), realibilityBroadcast, color='skyblue', edgecolor='black')
# 	# Add value labels on top of each bar
# 	for i, val in enumerate(realibilityBroadcast):
# 		plt.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
# 	plt.xlabel("Destination Node ID")
# 	plt.ylabel("Reliability of Broadcast Packets")
# 	plt.title("Broadcast Packet Delivery Reliability")
# 	plt.grid(axis='y')
# 	plt.tight_layout()
# 	plt.savefig(f"output/broadcast_reliability_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
# 	import pickle
# 	with open(f"output/broadcast_reliability_{conf.SELECTED_ROUTER_TYPE}.pkl", 'wb') as f:
# 	    pickle.dump(plt.gcf(), f)
# plt.close()


if source is not None:
	delaySensor = [0 for _ in range(N)]
	dest = source
	for source in range(N):
		if source != dest:
			if source in sensorPacketMeanDelays.keys():
				if dest in sensorPacketMeanDelays[source].keys():
					delaySensor[source] = sensorPacketMeanDelays[source][dest]
				else:
					delaySensor[source] = None
			else:
				delaySensor[source] = None
		else:
			delaySensor[source] = None
	delaySensorMatrix = np.array([val if val is not None else np.nan for val in delaySensor], dtype=float)

	plt.figure(figsize=(8, 6))
	bars = plt.bar(range(len(delaySensorMatrix)), delaySensorMatrix, color='skyblue', edgecolor='black')
	# Add value labels on top of each bar
	for i, val in enumerate(delaySensorMatrix):
		plt.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
	plt.xlabel("Source Node ID")
	plt.ylabel("Average Delay of Sensor Packets to Destination 0 (ms)")
	plt.title("Sensor Packet Delay to Destination 0")
	plt.grid(axis='y')
	plt.tight_layout()
	plt.savefig(f"output/sensor_delay_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
	import pickle
	with open(f"output/sensor_delay_{conf.SELECTED_ROUTER_TYPE}.pkl", 'wb') as f:
		pickle.dump(plt.gcf(), f)
plt.close()

delayDM = [[0 for _ in range(N)] for _ in range(N)]
for source in range(N):
	for dest in range(N):
		if source != dest:
			if source in dmPacketMeanDelays.keys():
				if dest in dmPacketMeanDelays[source].keys():
					delayDM[source][dest] = dmPacketMeanDelays[source][dest]
				else:
					delayDM[source][dest] = None
			else:
				delayDM[source][dest] = None
		else:
			delayDM[source][dest] = None
delayDMMatrix = np.array([[val if val is not None else np.nan for val in row] for row in delayDM], dtype=float)
np.fill_diagonal(delayDMMatrix, np.nan)
plt.figure(figsize=(8, 6))
plt.imshow(delayDMMatrix, cmap='YlGnBu', interpolation='nearest')
plt.colorbar(label='Average Delay (ms)')
for i in range(N):
	for j in range(N):
		if not np.isnan(delayDMMatrix[i][j]):
			plt.text(j, i, f"{delayDMMatrix[i][j]:.2f}", ha='center', va='center', color='black')
plt.title("DM Packet Delay Matrix")
plt.xlabel("Destination Node ID")
plt.ylabel("Source Node ID")
plt.xticks(ticks=np.arange(N), labels=np.arange(N))
plt.yticks(ticks=np.arange(N), labels=np.arange(N))
plt.grid(False)
plt.tight_layout()
plt.savefig(f"output/dm_delay_matrix_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
import pickle
with open(f"output/dm_delay_matrix_{conf.SELECTED_ROUTER_TYPE}.pkl", 'wb') as f:
    pickle.dump(plt.gcf(), f)
plt.close()

delayBroadcast = [0 for _ in range(N)]
source = 0
for dest in range(N):
	if source != dest:
		if source in brocastPacketMeanDelays.keys():
			if dest in brocastPacketMeanDelays[source].keys():
				delayBroadcast[dest] = brocastPacketMeanDelays[source][dest]
			else:
				delayBroadcast[dest] = None
		else:
			delayBroadcast[dest] = None
	else:
		delayBroadcast[dest] = None

delayBroadcastMatrix = np.array([val if val is not None else np.nan for val in delayBroadcast], dtype=float)
plt.figure(figsize=(8, 6))
bars = plt.bar(range(len(delayBroadcastMatrix)), delayBroadcastMatrix, color='skyblue', edgecolor='black')
# Add value labels on top of each bar
for i, val in enumerate(delayBroadcastMatrix):
	plt.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
plt.xlabel("Destination Node ID")
plt.ylabel("Average Delay of Broadcast Packets (ms)")
plt.title("Broadcast Packet Delay")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"output/broadcast_delay_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
import pickle
with open(f"output/broadcast_delay_{conf.SELECTED_ROUTER_TYPE}.pkl", 'wb') as f:
    pickle.dump(plt.gcf(), f)
plt.close()

extraSensorPacketsRatio = [0 for _ in range(N)]
dest = 0
for source in range(N):
	if source != dest:
		if source in SensorPacketsExtra.keys():
			if dest in SensorPacketsExtra[source].keys():
				extraSensorPacketsRatio[source] = SensorPacketsExtra[source][dest] / CreatedSensorPackets[source][dest] if CreatedSensorPackets[source][dest] > 0 else 0
			else:
				extraSensorPacketsRatio[source] = None
		else:
			extraSensorPacketsRatio[source] = None
	else:
		extraSensorPacketsRatio[source] = None
extraSensorPacketsRatio = np.array([val if val is not None else np.nan for val in extraSensorPacketsRatio], dtype=float)
plt.figure(figsize=(8, 6))
bars = plt.bar(range(len(extraSensorPacketsRatio)), extraSensorPacketsRatio, color='skyblue', edgecolor='black')
# Add value labels on top of each bar
for i, val in enumerate(extraSensorPacketsRatio):
	plt.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
plt.xlabel("Source Node ID")
plt.ylabel("Extra Sensor Packets Ratio")
plt.title("Extra Sensor Packets Ratio from Sources to Destination 0")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"output/extra_sensor_packets_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
import pickle
with open(f"output/extra_sensor_packets_{conf.SELECTED_ROUTER_TYPE}.pkl", 'wb') as f:
    pickle.dump(plt.gcf(), f)
plt.close()

extraDMPacketsRatio = [[0 for _ in range(N)] for _ in range(N)]
for source in range(N):
	for dest in range(N):
		if source != dest:
			if source in DMPacketsExtra.keys():
				if dest in DMPacketsExtra[source].keys():
					extraDMPacketsRatio[source][dest] = DMPacketsExtra[source][dest] / CreatedDMPackets[source][dest] if CreatedDMPackets[source][dest] > 0 else 0
				else:
					extraDMPacketsRatio[source][dest] = None
			else:
				extraDMPacketsRatio[source][dest] = None
		else:
			extraDMPacketsRatio[source][dest] = None
extraDMPacketsRatio = np.array([[val if val is not None else np.nan for val in row] for row in extraDMPacketsRatio], dtype=float)
np.fill_diagonal(extraDMPacketsRatio, np.nan)
plt.figure(figsize=(8, 6))
plt.imshow(extraDMPacketsRatio, cmap='YlGnBu', interpolation='nearest')
plt.colorbar(label='Extra DM Packets Ratio')
for i in range(N):
	for j in range(N):
		if not np.isnan(extraDMPacketsRatio[i][j]):
			plt.text(j, i, f"{extraDMPacketsRatio[i][j]:.2f}", ha='center', va='center', color='black')
plt.title("Extra DM Packets Ratio Matrix")
plt.xlabel("Destination Node ID")
plt.ylabel("Source Node ID")
plt.xticks(ticks=np.arange(N), labels=np.arange(N))
plt.yticks(ticks=np.arange(N), labels=np.arange(N))
plt.grid(False)
plt.tight_layout()
plt.savefig(f"output/extra_dm_packets_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
import pickle
with open(f"output/extra_dm_packets_{conf.SELECTED_ROUTER_TYPE}.pkl", 'wb') as f:
    pickle.dump(plt.gcf(), f)
plt.close()

extraBroadcastPacketsRatio = [0 for _ in range(N)]
source = 0
for dest in range(N):
	if source != dest:
		if source in BroadcastPacketsExtra.keys():
			extraBroadcastPacketsRatio[dest] = BroadcastPacketsExtra[dest] / TotalCreatedPackets if TotalCreatedPackets > 0 else 0
		else:
			extraBroadcastPacketsRatio[dest] = None
	else:
		extraBroadcastPacketsRatio[dest] = None
extraBroadcastPacketsRatio = np.array([val if val is not None else np.nan for val in extraBroadcastPacketsRatio], dtype=float)
plt.figure(figsize=(8, 6))
plt.bar(range(len(extraBroadcastPacketsRatio)), extraBroadcastPacketsRatio, color='skyblue', edgecolor='black')
# Add value labels on top of each bar
for i, val in enumerate(extraBroadcastPacketsRatio):
	plt.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
plt.xlabel("Destination Node ID")
plt.ylabel("Extra Broadcast Packets Ratio")
plt.title("Extra Broadcast Packets Ratio from Source 0")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"output/extra_broadcast_packets_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
import pickle
with open(f"output/extra_broadcast_packets_{conf.SELECTED_ROUTER_TYPE}.pkl", 'wb') as f:
    pickle.dump(plt.gcf(), f)
plt.close()

energyConsumedPerNode = {}
for n in nodes:
	energyConsumedPerNode[n.nodeid] = n.totalEnergyConsumedJ

# Convert to array for plotting
energyArray = np.array([energyConsumedPerNode.get(i, np.nan) for i in range(N)])

# Plot energy consumption
plt.figure(figsize=(8, 6))
bars = plt.bar(range(len(energyArray)), energyArray, color='skyblue', edgecolor='black')
# Add value labels on top of each bar
for i, val in enumerate(energyArray):
	if not np.isnan(val):
		plt.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
plt.xlabel("Node ID")
plt.ylabel("Energy Consumed (J)")
plt.title("Energy Consumption per Node")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"output/energy_consumption_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
with open(f"output/energy_consumption_{conf.SELECTED_ROUTER_TYPE}.pkl", 'wb') as f:
	pickle.dump(plt.gcf(), f)
plt.close()

# Save energy consumption to CSV
with open(f"output/energy_consumption_{conf.SELECTED_ROUTER_TYPE}.csv", mode="w", newline="") as f:
	writer = csv.writer(f)
	writer.writerow(["node_id", "energy_consumed_J"])
	for node_id, energy in sorted(energyConsumedPerNode.items()):
		writer.writerow([node_id, f"{energy:.4f}"])



def save_nested_dict_to_csv(data, filename):
    """
    Save a nested dictionary of the form:
    {outer: {inner: [values...]}} into a CSV file
    """
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "destination", "delay_value"])
        for outer, inner_dict in data.items():
            for inner, values in inner_dict.items():
                for v in values:
                    writer.writerow([outer, inner, v])

def save_reliability_matrix_to_csv(matrix, filename, node_ids=None):
    """
    Save a 2D reliability matrix (NxN numpy array) to CSV.
    Rows = source nodes, Columns = destination nodes
    
    Args:
        matrix: NxN numpy array with reliability values (0-1 or nan)
        filename: output CSV filename
        node_ids: optional list of node IDs (defaults to 0..N-1)
    """
    N = matrix.shape[0]
    if node_ids is None:
        node_ids = list(range(N))
    
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Header row: empty cell, then destination node IDs
        writer.writerow(["source\\dest"] + [f"dest_{nid}" for nid in node_ids])
        
        # Data rows: source node ID, then reliability values for each destination
        for i, src_id in enumerate(node_ids):
            row = [f"src_{src_id}"]
            for j in range(N):
                val = matrix[i][j]
                if np.isnan(val):
                    row.append("")  # Empty cell for nan
                else:
                    row.append(f"{val:.4f}")  # 4 decimal places
            writer.writerow(row)

def save_reliability_vector_to_csv(vector, filename, node_ids=None, column_name="reliability"):
    """
    Save a 1D reliability vector to CSV.
    
    Args:
        vector: 1D numpy array with reliability values (0-1 or nan)
        filename: output CSV filename
        node_ids: optional list of node IDs (defaults to 0..N-1)
        column_name: name for the reliability column
    """
    N = len(vector)
    if node_ids is None:
        node_ids = list(range(N))
    
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", column_name])
        
        for i, nid in enumerate(node_ids):
            val = vector[i]
            if np.isnan(val):
                writer.writerow([nid, ""])
            else:
                writer.writerow([nid, f"{val:.4f}"])

source = None
for node in nodes:
	if node.simRole == "Control_Center":
		source = node.nodeid
		break


if source is not None:
	print("Sensor packets delay arrays:", sensorPacketsDelayArrays)
	print("Broadcast packets delay arrays:", brocastPacketsDelayArrays)
	save_nested_dict_to_csv(sensorPacketsDelayArrays, f"output/sensor_packets_{conf.SELECTED_ROUTER_TYPE}.csv")
	save_nested_dict_to_csv(brocastPacketsDelayArrays, f"output/broadcast_packets_{conf.SELECTED_ROUTER_TYPE}.csv")
	# Save sensor reliability to CSV
	save_reliability_vector_to_csv(
    	realibilityMatrix, 
    	f"output/sensor_reliability_to_dest0_{conf.SELECTED_ROUTER_TYPE}.csv",
    	node_ids=list(range(N)),
    	column_name="reliability_to_dest_0")
	# Save broadcast reliability to CSV
	save_reliability_vector_to_csv(
    	realibilityBroadcast,
    	f"output/broadcast_reliability_{conf.SELECTED_ROUTER_TYPE}.csv",
    	node_ids=list(range(N)),
    	column_name="reliability_from_src_0")

print("DM packets delay arrays:", dmPacketsDelayArrays)
save_nested_dict_to_csv(dmPacketsDelayArrays, f"output/dm_packets_{conf.SELECTED_ROUTER_TYPE}.csv")



# Save DM reliability matrix to CSV
save_reliability_matrix_to_csv(
    DMmatrix,
    f"output/dm_reliability_matrix_{conf.SELECTED_ROUTER_TYPE}.csv",
    node_ids=list(range(N))
)


for n in nodes:
    with open(f'output/battery_node_{n.nodeid}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'battery_level_J'])  # Header
        for time, battery_level in sorted(n.batteryLevelByTime.items()):
            writer.writerow([time, battery_level])

os.makedirs('output/plots', exist_ok=True)

# Plot battery level over time for each node
for n in nodes:
	if n.batteryLevelByTime:
		times = sorted(n.batteryLevelByTime.keys())
		times_minutes = [t / 60000 for t in times]  # Convert ms to minutes
		battery_levels = [n.batteryLevelByTime[t] for t in times]
		
		plt.figure(figsize=(10, 6))
		plt.plot(times_minutes, battery_levels, linewidth=2, color='blue')
		plt.xlabel("Time (minutes)")
		plt.ylabel("Battery Level (J)")
		plt.title(f"Battery Level Over Time - Node {n.nodeid}")
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.savefig(f"output/plots/battery_node_{n.nodeid}_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
		plt.close()

# Plot all nodes' battery levels on one graph
plt.figure(figsize=(12, 7))
for n in nodes:
	if n.batteryLevelByTime:
		times = sorted(n.batteryLevelByTime.keys())
		times_minutes = [t / 60000 for t in times]  # Convert ms to minutes
		battery_levels = [n.batteryLevelByTime[t] for t in times]
		plt.plot(times_minutes, battery_levels, linewidth=1.5, label=f"Node {n.nodeid}", alpha=0.7)

plt.xlabel("Time (minutes)")
plt.ylabel("Battery Level (J)")
plt.title(f"Battery Levels Over Time - All Nodes ({conf.SELECTED_ROUTER_TYPE})")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"output/plots/battery_all_nodes_{conf.SELECTED_ROUTER_TYPE}.png", dpi=200, bbox_inches='tight')
plt.close()

print("\nSimulation complete.")