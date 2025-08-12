#!/usr/bin/env python3
import os
import sys
import random

import yaml
import simpy
import numpy as np

from lib import phy
from lib.common import Graph, plot_schedule, gen_scenario, run_graph_updates, setup_asymmetric_links
from lib.config import Config
from lib.discrete_event import BroadcastPipe
from lib.node import MeshNode
import matplotlib.pyplot as plt

VERBOSE = True
conf = Config()
random.seed(conf.SEED)


def verboseprint(*args, **kwargs):
	if VERBOSE:
		print(*args, **kwargs)


def parse_params(conf, args):
	# TODO: refactor with argparse
	if len(args) > 3:
		print("Usage: ./loraMesh [nr_nodes] [--from-file [file_name]]")
		print("Do not specify the number of nodes when reading from a file.")
		exit(1)
	else:
		if len(args) > 1:
			if isinstance(args[1], str) and ("--from-file" in args[1]):
				if len(args) > 2:
					string = args[2]
				else:
					string = 'nodeConfig.yaml'
				with open(os.path.join("out", string), 'r') as file:
					config = yaml.load(file, Loader=yaml.FullLoader)
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
print(delays)
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



# totalSensorPacketsCreated = sum([n.numberOfSensorPacketsCreated for n in nodes])

# totalSensorPacketsReceived = len(nodes[0].SensorPacketsReceived.keys()) # Node 0 is the only one that receives sensor packets

# extraSensorPackets = [nodes[0].SensorPacketsReceived[p] -1 for p in nodes[0].SensorPacketsReceived.keys() if nodes[0].SensorPacketsReceived[p] > 1]
# totalSensorPacketsAcked = sum([len(n.SensorPacketsAcked.keys()) for n in nodes])
# print("Total number of sensor packets created:", totalSensorPacketsCreated)
# print("Total number of sensor packets received:", totalSensorPacketsReceived)
# print("Total number of extra sensor packets:", sum(extraSensorPackets))
# print("Total number of sensor packets acked:", totalSensorPacketsAcked)

# print("reliability of sensor packets:", round(totalSensorPacketsReceived / totalSensorPacketsCreated * 100, 2), '%')

# print("Ratios of unwanted sensor packets:", round(sum(extraSensorPackets) / totalSensorPacketsCreated * 100, 2), '%')
# print("reliability of sensor packets acked:", round(totalSensorPacketsAcked / totalSensorPacketsCreated * 100, 2), '%')

# totalBroadcastPacketsCreated = sum([n.numberOfBroadcastPacketsCreated for n in nodes])
# totalBroadcastPacketsReceived = sum([len(n.BroadcastPacketsReceived.keys()) for n in nodes])
# totalBroadcastPacketsAverage = totalBroadcastPacketsReceived / (len(nodes) - 1) if len(nodes) > 1 else 0

# print("Total number of broadcast packets created:", totalBroadcastPacketsCreated)
# print("Total number of broadcast packets received:", totalBroadcastPacketsReceived)
# print("Average number of broadcast packets received per node (except CC):", round(totalBroadcastPacketsAverage, 2))
# print("reliability of broadcast packets:", round(totalBroadcastPacketsAverage / totalBroadcastPacketsCreated * 100, 2), '%')

# totalDMPacketsCreated = sum([n.numberOfDMPacketsCreated for n in nodes])
# totalDMPacketsReceived = sum([len(n.DMPacketsReceived.keys()) for n in nodes])
# extraDMPackets = [n.DMPacketsReceived[p]-1 for n in nodes for p in n.DMPacketsReceived.keys() if n.DMPacketsReceived[p] > 1]
# totalDMPacketsAcked = sum([len(n.DMPacketsAcked.keys()) for n in nodes])

# print("Total number of DM packets created:", totalDMPacketsCreated)
# print("Total number of DM packets received:", totalDMPacketsReceived)
# print("Total number of extra DM packets:", sum(extraDMPackets))
# print("Total number of DM packets acked:", totalDMPacketsAcked)
# print("reliability of DM packets:", round(totalDMPacketsReceived / totalDMPacketsCreated * 100, 2), '%')
# print("Ratios of unwanted DM packets:", round(sum(extraDMPackets) / totalDMPacketsCreated * 100, 2), '%')
# print("reliability of DM packets acked:", round(totalDMPacketsAcked / totalDMPacketsCreated * 100, 2), '%')

# SensorPacketsDelays =[]
# DMPacketsDelays = []
# ACKPacketsDelays = []

# for n in nodes:
#     SensorPacketsDelays.extend(n.SensorPacketsDelays)
#     DMPacketsDelays.extend(n.DMPacketsDelays)
#     ACKPacketsDelays.extend(n.ACKPacketsDelays)
# print(SensorPacketsDelays)
# meanSensorDelay = np.nanmean(SensorPacketsDelays) if SensorPacketsDelays else 0
# meanDMDelay = np.nanmean(DMPacketsDelays) if DMPacketsDelays else 0
# meanACKDelay = np.nanmean(ACKPacketsDelays) if ACKPacketsDelays else 0
# print("Average delay of sensor packets (ms):", round(meanSensorDelay, 2))
# print("Average delay of DM packets (ms):", round(meanDMDelay, 2))
# print("Average delay of ACK packets (ms):", round(meanACKDelay, 2))

sensorPacketDelaysAll = {}
dmPacketDelaysAll = {}
brocastPacketDelaysAll = {}

for n in nodes:
	senSorPacketsDelays = n.SensorPacketsDelays
	for originTxNodeId, delays in senSorPacketsDelays.items():
		if delays:
			if originTxNodeId not in sensorPacketDelaysAll:
				sensorPacketDelaysAll[originTxNodeId] = {}
			meanDelay = np.nanmean(delays)
			sensorPacketDelaysAll[originTxNodeId][n.nodeid] = meanDelay
			
			print(f"Average delay of sensor packets from node {originTxNodeId} to node {n.nodeid} (ms):", round(meanDelay, 2))

	broadCastPacketsDelays = n.BroadcastPacketsDelays
	for originTxNodeId, delays in broadCastPacketsDelays.items():
		if delays:
			if originTxNodeId not in brocastPacketDelaysAll:
				brocastPacketDelaysAll[originTxNodeId] = {}
			meanDelay = np.nanmean(delays)
			brocastPacketDelaysAll[originTxNodeId][n.nodeid] = meanDelay
			print(f"Average delay of broadcast packets from node {originTxNodeId} to node {n.nodeid} (ms):", round(meanDelay, 2))
	dmPacketsDelays = n.DMPacketsDelays
	for originTxNodeId, delays in dmPacketsDelays.items():
		if delays:
			if originTxNodeId not in dmPacketDelaysAll:
				dmPacketDelaysAll[originTxNodeId] = {}
			meanDelay = np.nanmean(delays)
			dmPacketDelaysAll[originTxNodeId][n.nodeid] = meanDelay
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

print("Number of DM packets created by each node:", CreatedDMPackets)
print("Number of DM packets received by each node:", RecivedDMPackets)
print("Number of sensor packets created by each node:", CreatedSensorPackets)
print("Number of sensor packets received by each node:", RecivedSensorPackets)
print("Total number of broadcast packets created by all nodes:", TotalCreatedPackets)
print("Number of broadcast packets received by each node:", RecivedBroadcastPackets)

print("Number of extra broadcast packets received by each node:", BroadcastPacketsExtra)
print("Number of extra DM packets received by each node:", DMPacketsExtra)
print("Number of extra sensor packets received by each node:", SensorPacketsExtra)

print("Sensor packets delays:", sensorPacketDelaysAll)
print("DM packets delays:", dmPacketDelaysAll)
print("Broadcast packets delays:", brocastPacketDelaysAll)

print("Range of nodes is",phy.MAXRANGE, "m")


N = len(nodes)
realiabilitySensor = [0 for _ in range(N)]
dest = 0
for source in range(N):
	if source != dest:
		if source in CreatedSensorPackets.keys(): 
			if dest in CreatedSensorPackets[source].keys():
				realiabilitySensor[source] = RecivedSensorPackets[source][dest] / CreatedSensorPackets[source][dest]
			else:
				realiabilitySensor[source] = None
		else:
			realiabilitySensor[source] = None


realibilityMatrix = np.array([val if val is not None else np.nan for val in realiabilitySensor], dtype=float)

print("Reliability of sensor packets from each node to node 0:", realibilityMatrix)

x = list(range(len(realibilityMatrix)))  # [0, 1, 2, 3, 4]

plt.figure(figsize=(8, 6))
bars = plt.bar(x, realibilityMatrix, color='skyblue', edgecolor='black')

# Add value labels on top of each bar
for i, val in enumerate(realibilityMatrix):
    plt.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

plt.xlabel("Source Node ID")
plt.ylabel("Reliability to Destination 0")
plt.title("Reliability from Sensors to Destination Node 0")
plt.xticks(x, [f"Src {i}" for i in x])
plt.grid(axis='y')
plt.tight_layout()
plt.show()

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
plt.show()

realibilityBroadcast = [0 for _ in range(N)]

source = 0
for dest in range(N):
	if source != dest:
		realibilityBroadcast[dest] = RecivedBroadcastPackets[dest] / TotalCreatedPackets if dest in RecivedBroadcastPackets else None
	else:
		realibilityBroadcast[dest] = None

realibilityBroadcast = np.array([val if val is not None else np.nan for val in realibilityBroadcast], dtype=float)

plt.figure(figsize=(8, 6))
bars = plt.bar(range(len(realibilityBroadcast)), realibilityBroadcast, color='skyblue', edgecolor='black')
# Add value labels on top of each bar
for i, val in enumerate(realibilityBroadcast):
	plt.text(i, val + 0.01, f"{val:.2f}", ha='center', va='bottom', fontsize=10)
plt.xlabel("Destination Node ID")
plt.ylabel("Reliability of Broadcast Packets")
plt.title("Broadcast Packet Delivery Reliability")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

delaySensor = [0 for _ in range(N)]
dest = 0
for source in range(N):
	if source != dest:
		if source in sensorPacketDelaysAll.keys():
			if dest in sensorPacketDelaysAll[source].keys():
				delaySensor[source] = sensorPacketDelaysAll[source][dest]
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
plt.show()

delayDM = [[0 for _ in range(N)] for _ in range(N)]
for source in range(N):
	for dest in range(N):
		if source != dest:
			if source in dmPacketDelaysAll.keys():
				if dest in dmPacketDelaysAll[source].keys():
					delayDM[source][dest] = dmPacketDelaysAll[source][dest]
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
plt.show()

delayBroadcast = [0 for _ in range(N)]
source = 0
for dest in range(N):
	if source != dest:
		if source in brocastPacketDelaysAll.keys():
			if dest in brocastPacketDelaysAll[source].keys():
				delayBroadcast[dest] = brocastPacketDelaysAll[source][dest]
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
plt.show()

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
plt.show()

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
plt.show()

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
plt.show()

if conf.PLOT:
	plot_schedule(conf, packets, messages)
