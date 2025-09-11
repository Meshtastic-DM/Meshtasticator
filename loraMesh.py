#!/usr/bin/env python3
import os
import sys
import random

import yaml
import simpy
import numpy as np

from lib.common import Graph, plot_schedule, gen_scenario, run_graph_updates, setup_asymmetric_links
from lib.config import Config
from lib.discrete_event import BroadcastPipe
from lib.node import MeshNode
from lib.packet import NODENUM_BROADCAST  # for interactive broadcast command

VERBOSE = True
conf = Config()
random.seed(conf.SEED)
np.random.seed(conf.SEED)

# ===================== INTERACTIVE MODE & COVERAGE =====================
# True  -> no random traffic; use terminal to send DM/BCAST & step time
# False -> original batch simulation that runs to SIMTIME
INTERACTIVE_MODE = True

# Default coverage radius (30 km)
COVERAGE_RADIUS_M = 30_000
# ======================================================================

# Make flags visible to nodes
conf.INTERACTIVE_MODE = INTERACTIVE_MODE


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


def print_final_stats(conf, nodes, packets, messages, delays,
					  totalPairs, symmetricLinks, asymmetricLinks, noLinks):
	print("\n====== END OF SIMULATION ======")
	print("*******************************")
	print(f"\nRouter Type: {conf.SELECTED_ROUTER_TYPE}")
	print('Number of messages created:', len(messages) if isinstance(messages, list) else "N/A")

	sent = len(packets)
	if conf.DMs:
		potentialReceivers = sent
	else:
		potentialReceivers = sent * (conf.NR_NODES - 1)
	print('Number of packets sent:', sent, 'to', potentialReceivers, 'potential receivers')

	nrCollisions = sum(1 for p in packets for n in nodes if p.collidedAtN[n.nodeid] is True)
	print("Number of collisions:", nrCollisions)
	nrSensed = sum(1 for p in packets for n in nodes if p.sensedByN[n.nodeid] is True)
	print("Number of packets sensed:", nrSensed)
	nrReceived = sum(1 for p in packets for n in nodes if p.receivedAtN[n.nodeid] is True)
	print("Number of packets received:", nrReceived)

	meanDelay = np.nanmean(delays) if len(delays) else float('nan')
	print('Delay average (ms):', round(meanDelay, 2) if meanDelay == meanDelay else "nan")

	txAirUtilization = (sum(n.txAirUtilization for n in nodes) / conf.NR_NODES / conf.SIMTIME * 100) if conf.SIMTIME > 0 else 0.0
	print('Average Tx air utilization:', round(txAirUtilization, 2), '%')

	if nrSensed != 0:
		collisionRate = float(nrCollisions) / nrSensed
		print("Percentage of packets that collided:", round(collisionRate * 100, 2))
	else:
		print("No packets sensed.")

	msg_count = len(messages)
	nodeReach = (sum(n.usefulPackets for n in nodes) / (msg_count * (conf.NR_NODES - 1))) if (msg_count and conf.NR_NODES > 1) else 0.0
	print("Average percentage of nodes reached:", round(nodeReach * 100, 2))
	if nrReceived != 0:
		usefulness = sum(n.usefulPackets for n in nodes) / nrReceived
		print("Percentage of received packets containing new message:", round(usefulness * 100, 2))
	else:
		print('No packets received.')

	delayDropped = sum(n.droppedByDelay for n in nodes)
	print("Number of packets dropped by delay/hop limit:", delayDropped)

	if conf.MODEL_ASYMMETRIC_LINKS:
		print("Asymmetric links:", round(asymmetricLinks / totalPairs * 100, 2), '%')
		print("Symmetric links:", round(symmetricLinks / totalPairs * 100, 2), '%')
		print("No links:", round(noLinks / totalPairs * 100, 2), '%')

	if conf.MOVEMENT_ENABLED:
		movingNodes = sum(1 for n in nodes if n.isMoving is True)
		print("Number of moving nodes:", movingNodes)
		gpsEnabled = sum(1 for n in nodes if n.gpsEnabled is True)
		print("Number of moving nodes w/ GPS:", gpsEnabled)


# -------- Coverage helpers --------
def start_coverage_tracking(conf, nodes, src_id, seq, radius_m=COVERAGE_RADIUS_M):
	"""Initialize coverage tracking for message (src_id, seq)."""
	conf.COV_ACTIVE = True
	conf.COV_SRC = src_id
	conf.COV_SEQ = seq
	conf.COV_SEND_T = conf._ENV.now if hasattr(conf, "_ENV") else 0
	conf.COV_RADIUS_M = radius_m

	# Targets: nodes within radius of map center (conf.OX, conf.OY), excluding source
	r2 = radius_m * radius_m
	targets = set()
	for i, n in enumerate(nodes):
		if i == src_id:
			continue
		dx = n.x - conf.OX
		dy = n.y - conf.OY
		if (dx*dx + dy*dy) <= r2:
			targets.add(i)

	conf.COV_TARGET_IDS = targets
	conf.COV_FIRST_RX = {i: None for i in targets}


def coverage_report(conf):
	if not getattr(conf, "COV_ACTIVE", False):
		print("Coverage not active. Send a broadcast (bcast) to start tracking.")
		return

	targets = conf.COV_TARGET_IDS
	first_rx = conf.COV_FIRST_RX
	got = [t for t in targets if first_rx[t] is not None]
	miss = [t for t in targets if first_rx[t] is None]
	coverage_pct = (100.0 * len(got) / max(1, len(targets))) if targets else 0.0

	times = sorted((first_rx[t] - conf.COV_SEND_T) for t in got)
	def pct(arr, q):
		if not arr:
			return float('nan')
		idx = max(0, min(len(arr)-1, int(q * len(arr)) - 1))
		return arr[idx]

	print("\n====== COVERAGE (R <= {} m) ======".format(conf.COV_RADIUS_M))
	print(f"Origin: node {conf.COV_SRC}, seq {conf.COV_SEQ}, sent @ {conf.COV_SEND_T} ms")
	print(f"Targets in radius: {len(targets)}")
	print(f"Reached: {len(got)}  |  Missed: {len(miss)}  |  Coverage: {coverage_pct:.2f}%")
	if times:
		print(f"P50: {pct(times, 0.50):.1f} ms   P90: {pct(times, 0.90):.1f} ms   P99: {pct(times, 0.99):.1f} ms   Max: {times[-1]:.1f} ms")
	else:
		print("No targets have received the message yet.")

	if miss:
		# show a few missing nodes
		show = ", ".join(str(i) for i in list(miss)[:10])
		etc = " ..." if len(miss) > 10 else ""
		print(f"Missing nodes (first 10): {show}{etc}")

	if len(got) == len(targets) and times:
		print(f"Time to reach ALL targets: {times[-1]:.1f} ms")






def repl(env, conf, nodes, packets, messages, delays):
    def print_help():
        print("""
Commands:
  bcast  <src> [ack=0|1] [radius_m] - broadcast from <src>; if radius_m is given, (re)start 30km-coverage tracking with that radius
  dm     <src> <dst> [ack=1]        - send DM from <src> to <dst>
  cov                               - show coverage stats for last tracked broadcast
  addg   <N> [radius_m=30000] [sigma_m=radius/3] - add N nodes in a Gaussian cluster (truncated) around the map center
  step   <ms>                       - advance simulation by <ms> milliseconds
  time                               - show current sim time (ms)
  nodes                              - list node ids and positions
  stats                              - print current stats snapshot
  help                               - show this help
  quit / exit                        - end the simulation
""")

    def do_stats():
        print("---- stats ----")
        print(f"t = {env.now} ms")
        print(f"messages: {len(messages)}")
        sent = len(packets)
        print(f"packets sent: {sent}")
        nrCollisions = sum(1 for p in packets for n in nodes if p.collidedAtN[n.nodeid])
        nrSensed     = sum(1 for p in packets for n in nodes if p.sensedByN[n.nodeid])
        nrReceived   = sum(1 for p in packets for n in nodes if p.receivedAtN[n.nodeid])
        print(f"collisions: {nrCollisions} | sensed: {nrSensed} | received: {nrReceived}")
        if delays:
            print(f"avg delay: {np.nanmean(delays):.2f} ms (n={len(delays)})")

    # ---- local helpers for Gaussian placement and node creation ----
    def _sample_gaussian_points_2d(n, radius_m, ox, oy, sigma_m=None, max_tries=1000):
        if sigma_m is None:
            sigma_m = radius_m / 3.0
        pts = []
        tries = 0
        r2 = radius_m * radius_m
        while len(pts) < n and tries < max_tries:
            tries += 1
            x = np.random.normal(loc=ox, scale=sigma_m)
            y = np.random.normal(loc=oy, scale=sigma_m)
            dx = x - ox
            dy = y - oy
            if (dx * dx + dy * dy) <= r2:
                pts.append((float(x), float(y)))
        # fallback to uniform-in-disk if rejection didn't fill up
        while len(pts) < n:
            u = np.random.random()
            theta = 2 * np.pi * np.random.random()
            r = radius_m * np.sqrt(u)
            pts.append((float(ox + r * np.cos(theta)), float(oy + r * np.sin(theta))))
        return pts

    def _add_nodes_gaussian(n_new, radius_m, sigma_m=None, z_default=1.0):
        if n_new <= 0:
            return 0
        # sample positions
        pts = _sample_gaussian_points_2d(n_new, radius_m, conf.OX, conf.OY, sigma_m=sigma_m)
        created = 0
        for (x, y) in pts:
            packetsAtN.append([])  # prepare per-node bucket before constructing the node
            node_id = len(nodes)
            node = MeshNode(conf, nodes, env, bc_pipe, node_id, conf.PERIOD,
                            messages, packetsAtN, packets, delays, None, messageSeq, verboseprint)
            node.x = x
            node.y = y
            node.z = z_default

            # force it to be a muted client
            node.isClientMute = True
            node.isRouter = False
            node.isRepeater = False

            nodes.append(node)
            graph.add_node(node)
            created += 1
        # sync count and recompute links
        conf.NR_NODES = len(nodes)
        setup_asymmetric_links(conf, nodes)
        return created

    print("\n====== INTERACTIVE SIM ======")
    print_help()

    # tiny warmup to avoid divide-by-zero if MAC uses env.now
    env.run(until=max(1, env.now))

    while True:
        try:
            line = input("sim> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        toks = line.split()
        cmd = toks[0].lower()

        if cmd in ("quit", "exit"):
            break
        elif cmd == "help":
            print_help()
        elif cmd == "time":
            print(f"t = {env.now} ms")
        elif cmd == "nodes":
            for i, n in enumerate(nodes):
                print(f"#{i}: ({n.x:.1f},{n.y:.1f}) z={n.z} router={n.isRouter} repeater={n.isRepeater}")
        elif cmd == "stats":
            do_stats()
        elif cmd == "cov":
            coverage_report(conf)
        elif cmd == "step":
            if len(toks) < 2:
                print("usage: step <ms>")
                continue
            try:
                ms = int(float(toks[1]))
            except ValueError:
                print("ms must be a number")
                continue
            env.run(until=env.now + ms)
            print(f"advanced to t={env.now} ms")
        elif cmd in ("bcast", "broadcast"):
            if len(toks) < 2:
                print("usage: bcast <src> [ack=0|1] [radius_m]")
                continue
            try:
                src = int(toks[1])
                ack = bool(int(toks[2])) if len(toks) >= 3 else False
                rad = int(toks[3]) if len(toks) >= 4 else COVERAGE_RADIUS_M
            except ValueError:
                print("bad args")
                continue
            if not (0 <= src < len(nodes)):
                print("invalid src id")
                continue
            # Send the broadcast
            p = nodes[src].send_packet(NODENUM_BROADCAST, type="BCAST")
            try:
                p.wantAck = ack
            except Exception:
                pass
            # Start (or restart) coverage tracking for this broadcast's (src, seq)
            conf._ENV = env  # allow tracker to read env.now
            start_coverage_tracking(conf, nodes, src, p.seq, rad)
            # advance enough for a TX/RX round to complete; you can step more as needed
            env.run(until=env.now + 5000)
            print(f"broadcast from {src} done; t={env.now} ms")
            coverage_report(conf)
        elif cmd == "dm":
            if len(toks) < 3:
                print("usage: dm <src> <dst> [ack=1]")
                continue
            try:
                src = int(toks[1]); dst = int(toks[2])
                ack = bool(int(toks[3])) if len(toks) >= 4 else True
            except ValueError:
                print("bad args")
                continue
            if not (0 <= src < len(nodes)) or not (0 <= dst < len(nodes)) or src == dst:
                print("invalid src/dst")
                continue
            p = nodes[src].send_packet(dst, type="DM")
            try:
                p.wantAck = ack
            except Exception:
                pass
            env.run(until=env.now + 5000)
            print(f"dm {src}->{dst} sent; t={env.now} ms")
        elif cmd == "addg":
            if len(toks) < 2:
                print("usage: addg <N> [radius_m=30000] [sigma_m=radius/3]")
                continue
            try:
                N = int(toks[1])
                radius_m = int(toks[2]) if len(toks) >= 3 else COVERAGE_RADIUS_M
                sigma_m = float(toks[3]) if len(toks) >= 4 else None
            except ValueError:
                print("bad args: N must be int; radius_m int; sigma_m float")
                continue
            before = len(nodes)
            made = _add_nodes_gaussian(N, radius_m, sigma_m=sigma_m)
            after = len(nodes)
            print(f"added {made} nodes; total nodes: {after} (was {before})")
        else:
            print("unknown command. type 'help'.")




# ======================== main ========================

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

if not INTERACTIVE_MODE:
	# -------- original batch run ----------
	print("\n====== START OF SIMULATION ======")
	env.run(until=conf.SIMTIME)
	print_final_stats(conf, nodes, packets, messages, delays,
					 totalPairs, symmetricLinks, asymmetricLinks, noLinks)
	graph.save()
	if conf.PLOT:
		plot_schedule(conf, packets, messages)
else:
	# -------- interactive terminal ----------
	# Tell coverage helpers where to read current time from
	conf._ENV = env
	repl(env, conf, nodes, packets, messages, delays)
	print("\n====== END (interactive) ======")
	graph.save()
	if conf.PLOT:
		plot_schedule(conf, packets, messages)
