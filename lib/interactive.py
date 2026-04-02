import cmd
import socket
import sys
import threading
import time
import yaml
import os
from shutil import which
import subprocess


import google.protobuf.json_format as proto
from matplotlib import patches
from meshtastic import tcp_interface, serial_interface, BROADCAST_NUM, mesh_pb2, admin_pb2, telemetry_pb2, portnums_pb2, channel_pb2
from pubsub import pub
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

from lib.config import Config
import lib.phy as phy
from lib.common import calc_dist, gen_scenario, find_random_position, Graph

conf = Config()
HW_ID_OFFSET = 16
TCP_PORT_OFFSET = 4403
TCP_PORT_CLIENT = 4402
MAX_TO_FROM_RADIO_SIZE = 512
DEVICE_SIM_DOCKER_IMAGE = "meshtastic/device-simulator"
MESHTASTICD_PATH_DOCKER = "./meshtasticd"


class InteractiveNode:
    def __init__(self, nodes, nodeId, hwId, TCPPort, nodeConfig):
        self.nodeid = nodeId
        if nodeConfig is not None:
            self.x, self.y, self.z = nodeConfig['x'], nodeConfig['y'], nodeConfig['z']
            self.isRouter = nodeConfig['isRouter']
            self.isRepeater = nodeConfig['isRepeater']
            self.isClientMute = nodeConfig['isClientMute']
            self.hopLimit = nodeConfig['hopLimit']
            self.antennaGain = nodeConfig['antennaGain']
            self.neighborInfo = nodeConfig['neighborInfo']
        else:
            self.x, self.y = find_random_position(conf, nodes)
            self.z = conf.HM
            self.isRouter = conf.router
            self.isRepeater = False
            self.isClientMute = False
            self.hopLimit = conf.hopLimit
            self.antennaGain = conf.GL
            self.neighborInfo = False
        self.iface = None
        self.hwId = hwId
        self.TCPPort = TCPPort
        self.timestamps = []
        self.channelUtilization = []
        self.airUtilTx = []
        self.numPacketsTx = 0
        self.numPacketsRx = 0
        self.numPacketsRxBad = 0
        self.numRxDupe = 0
        self.numTxRelay = 0
        self.numTxRelayCanceled = 0
        self.desiredRegion = "IN"
        self.desiredModemPreset = "SHORT_TURBO"

    def add_interface(self, iface):
        self.iface = iface

    def set_config(self):
        requiresReboot = False
        # Set a long and short name
        p = admin_pb2.AdminMessage()
        p.set_owner.long_name = "Node "+str(self.nodeid)
        p.set_owner.short_name = str(self.nodeid)
        self.iface.localNode._sendAdmin(p)
        
        # Set LoRa region (REQUIRED for PKI key generation!)
        loraConfig = self.iface.localNode.localConfig.lora

        # Resolve enum values by name from protobuf descriptor (no hardcoded numeric map)
        region_name = (self.desiredRegion or "IN").upper()
        modem_name = (self.desiredModemPreset or "SHORT_TURBO").upper()

        region_enum = loraConfig.DESCRIPTOR.fields_by_name["region"].enum_type
        modem_enum = loraConfig.DESCRIPTOR.fields_by_name["modem_preset"].enum_type

        if region_name in region_enum.values_by_name:
            setattr(loraConfig, "region", region_enum.values_by_name[region_name].number)
        else:
            print(f"[warn] Unknown region '{region_name}', keeping current region")

        if modem_name in modem_enum.values_by_name:
            setattr(loraConfig, "modem_preset", modem_enum.values_by_name[modem_name].number)
        else:
            print(f"[warn] Unknown modem preset '{modem_name}', keeping current preset")

        if self.hopLimit != 3:
            setattr(loraConfig, 'hop_limit', self.hopLimit)

        p = admin_pb2.AdminMessage()
        p.set_config.lora.CopyFrom(loraConfig)
        self.iface.localNode._sendAdmin(p)
        requiresReboot = True  # Region change requires reboot for PKI key generation

        if self.isRouter:
            requiresReboot = True
            deviceConfig = self.iface.localNode.localConfig.device
            setattr(deviceConfig, 'role', "ROUTER")
            p = admin_pb2.AdminMessage()
            p.set_config.device.CopyFrom(deviceConfig)
            self.iface.localNode._sendAdmin(p)
        elif self.isRepeater:
            requiresReboot = True
            deviceConfig = self.iface.localNode.localConfig.device
            setattr(deviceConfig, 'role', "REPEATER")
            p = admin_pb2.AdminMessage()
            p.set_config.device.CopyFrom(deviceConfig)
            self.iface.localNode._sendAdmin(p)
        elif self.isClientMute:
            requiresReboot = True
            deviceConfig = self.iface.localNode.localConfig.device
            setattr(deviceConfig, 'role', "CLIENT_MUTE")
            p = admin_pb2.AdminMessage()
            p.set_config.device.CopyFrom(deviceConfig)
            self.iface.localNode._sendAdmin(p)

        if self.neighborInfo:
            requiresReboot = True
            moduleConfig = self.iface.localNode.moduleConfig.neighbor_info
            setattr(moduleConfig, 'enabled', 1)
            setattr(moduleConfig, 'update_interval', 30)
            p = admin_pb2.AdminMessage()
            p.set_module_config.neighbor_info.CopyFrom(moduleConfig)
            self.iface.localNode._sendAdmin(p)

        base_lat = 44
        base_lon = -105
        conv_factor = 0.0001
        lat = base_lat + (self.y * conv_factor)
        lon = base_lon + (self.x * conv_factor)
        self.iface.sendPosition(lat, lon, 0)

        return requiresReboot

    def add_admin_channel(self):
        ch = self.iface.localNode.getChannelByChannelIndex(1)
        chs = channel_pb2.ChannelSettings()
        chs.psk = b'\xb0X\xad\xb3\xa5\xd0?$\x8c\x92{\xcd^(\xeb\xb7\x01\x84"\xc9\xf4\x06:\x8d\xfdD#\x08\xe5\xc2\xd7\xdc'
        chs.name = "admin"
        ch.settings.CopyFrom(chs)
        ch.role = channel_pb2.Channel.Role.SECONDARY
        self.iface.localNode.channels[ch.index] = ch
        self.iface.localNode.writeChannel(ch.index)
        time.sleep(1)


class InteractivePacket:

    def __init__(self, packet, id):
        self.packet = packet
        self.localId = id

    def setTxRxs(self, transmitter, receivers):
        self.transmitter = transmitter
        self.receivers = receivers

    def setRSSISNR(self, rssis, snrs):
        self.rssis = rssis
        self.snrs = snrs


class InteractiveGraph(Graph):
    def __init__(self):
        super().__init__(conf)
        self.routes = False

    def init_routes(self, sim):
        if not sim.docker:
            sim.close_nodes()
        if not self.routes:
            self.routes = True
            self.sim = sim
            self.arrows = []
            self.txts = []
            self.annots = []
            self.firstTime = True
            self.defaultHopLimit = conf.hopLimit
            self.fig.subplots_adjust(bottom=0.2)
            axbox = self.fig.add_axes([0.5, 0.04, 0.1, 0.06])
            self.text_box = TextBox(axbox, "Message ID: ", initial="0")
            self.text_box.disconnect("button_press_event")
            self.text_box.on_submit(self.submit)
            self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            self.fig.canvas.mpl_connect("close_event", self.on_close)
            print("On the scenario plot, enter a message ID to show its route. Close the figure to exit.")
            self.fig.canvas.draw_idle()
            self.fig.canvas.get_tk_widget().focus_set()
            plt.show()
        elif sim.docker:
            sim.close_nodes()

    def clear_route(self):
        for arr in self.arrows.copy():
            arr.remove()
            self.arrows.remove(arr)
        for ann in self.annots.copy():
            ann.remove()
            self.annots.remove(ann)

    def plot_route(self, messageId):
        if self.firstTime:
            print('Hover over an arc to show some info and click to remove it afterwards.')
            print('Close the window to exit the simulator.')
        self.firstTime = False
        packets = [p for p in self.packets if p.localId == messageId]
        if len(packets) > 0:
            self.clear_route()
            style = "Simple, tail_width=0.5, head_width=4, head_length=8"
            pairs = dict.fromkeys(list(set(p.transmitter for p in packets)), [])
            for p in packets:
                sim_port = p.packet.get("decoded", {}).get("simulator", {}).get("portnum", "ENCRYPTED")
                tx = p.transmitter
                rxs = p.receivers
                rxCnt = 1
                for ri, rx in enumerate(rxs):
                    # calculate how many packets with the same Tx and Rx we have
                    found = False
                    for pi, rxPair in enumerate(pairs.get(tx)):  # pair is rx.nodeid and its count for this transmitter
                        if rxPair[0] == rx.nodeid:
                            found = True
                            rxCnt = rxPair[1] + 1
                            updated = pairs.get(tx).copy()
                            updated[pi] = (rx.nodeid, rxCnt)
                            pairs.update({tx: updated})
                    if not found:
                        rxCnt = 1
                        pairs.get(tx).append((rx.nodeid, rxCnt))
                    kw = dict(arrowstyle=style, color=plt.cm.Set1(tx.nodeid))
                    rad = str(rxCnt*0.1)  # set the rad to Tx-Rx pair count
                    patch = patches.FancyArrowPatch((tx.x, tx.y), (rx.x, rx.y), connectionstyle="arc3,rad="+rad, **kw)
                    self.ax.add_patch(patch)

                    if int(p.packet["to"]) == BROADCAST_NUM:
                        to = "All"
                    else:
                        to = str(p.packet["to"]-HW_ID_OFFSET)

                    if p.packet["from"] == tx.hwId:
                        if "requestId" in p.packet["decoded"]:
                            if p.packet["priority"] == "ACK":
                                msgType = "Real\/ACK"
                            else:
                                msgType = "Response"
                        else:
                            msgType = "Original message"
                    elif "requestId" in p.packet["decoded"]:
                        if sim_port == "ROUTING_APP":
                            msgType = "Forwarding\/real\/ACK"
                        else:
                            msgType = "Forwarding\/response"
                    else:
                        if int(p.packet['from']) == rx.hwId:
                            msgType = "Implicit\/ACK"
                        else:
                            if to == "All":
                                msgType = "Rebroadcast"
                            else:
                                msgType = "Forwarding\/message"

                    hopLimit = p.packet.get("hopLimit")

                    fields = [ r"$\bf{" + msgType + "}$"
                             , f"Original sender: {p.packet['from'] - HW_ID_OFFSET}"
                             , f"Destination: {to}"
                             , f"Portnum: {sim_port}"
                             , f"HopLimit: {hopLimit}" if hopLimit else ""
                             , f"RSSI: {round(p.rssis[ri], 2)} dBm"
                             ]
                    table = "\n".join(filter(None, fields))
                    annot = self.ax.annotate(table, xy=((tx.x+rx.x)/2, rx.y+150), bbox=dict(boxstyle="round", fc="w"))
                    annot.get_bbox_patch().set_facecolor(patch.get_facecolor())
                    annot.get_bbox_patch().set_alpha(0.4)
                    annot.set_visible(False)
                    self.arrows.append(patch)
                    self.annots.append(annot)
            self.fig.canvas.draw_idle()
            self.fig.suptitle('Route of message '+str(messageId)+' and ACKs')
        else:
            print('Could not find message ID.')

    def hover(self, event):
        if event.inaxes == self.ax:
            for i, a in enumerate(self.arrows):
                annot = self.annots[i]
                cont, _ = a.contains(event)
                if cont:
                    annot.set_visible(True)
                    self.fig.canvas.draw()
                    break

    def on_click(self, event):
        for annot in self.annots:
            if annot.get_visible():
                annot.set_visible(False)
                self.fig.canvas.draw_idle()

    def on_close(self, event):
        plt.close('all')

    def submit(self, val):
        messageId = int(val)
        self.plot_route(messageId)

    def plot_metrics(self, nodes):
        if any(len(n.timestamps) > 1 for n in nodes):
            plt.figure()
            for n in nodes:
                if len(n.timestamps) > 0:
                    initTime = n.timestamps[0]
                    plt.plot([t-initTime for t in n.timestamps], n.channelUtilization, label=str(n.nodeid), marker=".")
            plt.ylabel('Channel utilization (%)')
            plt.xlabel('Time (s)')
            plt.legend(title='Node ID')
            plt.figure()
            for n in nodes:
                if len(n.timestamps) > 0:
                    initTime = n.timestamps[0]
                    plt.plot([t-initTime for t in n.timestamps], n.airUtilTx, label=str(n.nodeid), marker=".")
            plt.ylabel('Hourly Tx air utilization (%)')
            plt.xlabel('Time (s)')
            plt.legend(title='Node ID')

        if any(n.numPacketsRxBad > 0 for n in nodes):  # Only really interesting if there are bad packets (meaning collisions)
            stats = ['Tx', 'Rx', 'Rx bad', 'Rx dupe', 'Tx relay', 'Tx relay canceled']
            num_stats = len(stats)
            num_nodes = len(nodes)
            x = np.arange(num_stats)
            _, ax = plt.subplots(figsize=(12, 6))
            data = [[n.numPacketsTx for n in nodes], [n.numPacketsRx for n in nodes], [n.numPacketsRxBad for n in nodes], [n.numRxDupe for n in nodes], [n.numTxRelay for n in nodes], [n.numTxRelayCanceled for n in nodes]]
            bar_width = 0.15
            for i in range(num_nodes):
                x_positions = x + (i - (num_nodes / 2)) * bar_width + bar_width / 2
                ax.bar(x_positions, [row[i] for row in data], width=bar_width, label=nodes[i].nodeid)
                ax.set_xticks([])
            ax.set_ylabel('Number of packets')
            ax.set_xticks(x)
            ax.set_xticklabels(stats)
            ax.legend(title='Node ID')
            ax.set_title('Packet statistics')


class InteractiveSim:
    def __init__(self, args):
        self.messages = []
        self.messageId = -1
        self.nodes = []
        foundNodes = False
        self.clientConnected = False
        self.forwardSocket = None
        self.clientSocket = None
        self.nodeThread = None
        self.clientThread = None
        self.wantExit = False

        # argument handling
        self.script = args.script
        self.docker = args.docker
        self.forwardToClient = args.forward
        self.emulateCollisions = args.collisions
        self.removeConfig = not args.from_file

        # NEW: bridge + radio profile settings
        self.serialPort = args.serial
        self.mirrorNode = int(args.mirror_node)
        self.region = (args.region or "IN").upper()
        self.modemPreset = (args.modem_preset or "SHORT_TURBO").upper()
        self.serial_iface = None

        # NEW: loop suppression
        self._seen_from_serial = {}
        self._seen_to_serial = {}
        self._seen_ttl_s = 30.0

        if args.from_file:
            foundNodes = True
            with open(os.path.join("out", "nodeConfig.yaml"), 'r') as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
            conf.NR_NODES = len(config.keys())
        elif args.nrNodes > 0:
            conf.NR_NODES = args.nrNodes
            foundNodes = True
            config = [None for _ in range(conf.NR_NODES)]
        if not foundNodes:
            print("nrNodes was not specified, generating scenario...")
            config = gen_scenario(conf)
            conf.NR_NODES = len(config.keys())

        if not self.docker and not sys.platform.startswith('linux'):
            print("Docker is required for non-Linux OS.")
            self.docker = True

        self.graph = InteractiveGraph()
        for n in range(conf.NR_NODES):
            node = InteractiveNode(self.nodes, n, self.node_id_to_hw_id(n), n + TCP_PORT_OFFSET, config[n])
            node.desiredRegion = self.region
            node.desiredModemPreset = self.modemPreset
            self.nodes.append(node)
            self.graph.add_node(node)

        if self.serialPort is not None and not (0 <= self.mirrorNode < len(self.nodes)):
            raise ValueError(f"--mirror-node must be in [0, {len(self.nodes)-1}]")

        print("Booting nodes...")

        self.init_nodes(args)
        iface0 = self.init_forward()
        self.init_communication(iface0)

    # NEW
    def _gc_seen(self):
        now = time.time()
        for table in (self._seen_from_serial, self._seen_to_serial):
            stale = [k for k, ts in table.items() if now - ts > self._seen_ttl_s]
            for k in stale:
                del table[k]

    # NEW
    def _seen_recent(self, table, pkt_id):
        if pkt_id is None:
            return False
        ts = table.get(pkt_id)
        return ts is not None and (time.time() - ts) <= self._seen_ttl_s

    # NEW
    def _mark_seen(self, table, pkt_id):
        if pkt_id is not None:
            table[pkt_id] = time.time()

    # NEW
    def init_serial_bridge(self):
        if not self.serialPort:
            return
        try:
            self.serial_iface = serial_interface.SerialInterface(devPath=self.serialPort)
            self.nodes[self.mirrorNode].add_interface(self.serial_iface)
            print(f"[serial] connected on {self.serialPort}; mirror node={self.mirrorNode}")
            pub.subscribe(self.on_receive_serial, "meshtastic.receive")
        except Exception as ex:
            self.serial_iface = None
            print(f"[serial] failed to open {self.serialPort}: {ex}")

    def init_nodes(self, args):
        run_nodes = [n for n in self.nodes if not (self.serialPort and n.nodeid == self.mirrorNode)]
        if len(run_nodes) == 0:
            print("[init_nodes] no virtual/native nodes to spawn (serial-only mirror mode).")
            return

        if self.docker:
            try:
                import docker
            except ImportError:
                print("Please install the Docker SDK for Python with 'pip3 install docker'.")
                exit(1)

            n0 = run_nodes[0]
            dockerClient = docker.from_env()
            startNode = f"{MESHTASTICD_PATH_DOCKER} "
            if self.removeConfig:
                startNode += "-e "

            if sys.platform == "darwin":
                self.container = dockerClient.containers.run(
                    DEVICE_SIM_DOCKER_IMAGE,
                    f"{startNode} -d /home/node{n0.nodeid} -h {n0.hwId} -p {n0.TCPPort}",
                    ports=dict(zip((f'{n.TCPPort}/tcp' for n in run_nodes), (n.TCPPort for n in run_nodes))),
                    name="Meshtastic", detach=True, auto_remove=True, user="root"
                )
                for n in run_nodes[1:]:
                    if self.emulateCollisions:
                        time.sleep(2)
                    self.container.exec_run(f"{startNode} -d /home/node{n.nodeid} -h {n.hwId} -p {n.TCPPort}", detach=True, user="root")
            else:
                self.container = dockerClient.containers.run(
                    DEVICE_SIM_DOCKER_IMAGE,
                    f"sh -c '{startNode} -d /home/node{n0.nodeid} -h {n0.hwId} -p {n0.TCPPort} > /home/out_{n0.nodeid}.log'",
                    ports=dict(zip((f'{n.TCPPort}/tcp' for n in run_nodes), (n.TCPPort for n in run_nodes))),
                    name="Meshtastic", detach=True, auto_remove=True, user="root",
                    volumes={"Meshtasticator": {'bind': '/home/', 'mode': 'rw'}}
                )
                for n in run_nodes[1:]:
                    if self.emulateCollisions:
                        time.sleep(2)
                    self.container.exec_run(f"sh -c '{startNode} -d /home/node{n.nodeid} -h {n.hwId} -p {n.TCPPort} > /home/out_{n.nodeid}.log'", detach=True, user="root")
        else:
            os.makedirs("out", exist_ok=True)
            prog = os.path.join(args.program, 'program')
            for n in run_nodes:
                node_args = (
                    f"-d {os.path.expanduser('~')}/.portduino/node{n.nodeid} "
                    f"-h {n.hwId} "
                    f"-p {n.TCPPort} "
                    + ("-e " if self.removeConfig else "")
                )
                cmd = f"stdbuf -oL -eL {prog} {node_args} 2>&1 | tee out/node{n.nodeid}.log"
                if which('gnome-terminal') is not None:
                    os.system(f"gnome-terminal --title='Node {n.nodeid}' -- bash -lc \"{cmd}\"")
                elif which('xterm') is not None:
                    os.system(f"xterm -title 'Node {n.nodeid}' -e bash -lc \"{cmd}\"")
                else:
                    print('Native mode requires gnome-terminal or xterm (WSLg/X11).')
                    exit(1)
                if self.emulateCollisions and n.nodeid != run_nodes[-1].nodeid:
                    time.sleep(2)

    def init_forward(self):
        if self.forwardToClient:
            self.forwardSocket = socket.socket()
            self.forwardSocket.bind(('', TCP_PORT_CLIENT))
            self.forwardSocket.listen()
            print(f"Please connect the client to TCP port {TCP_PORT_CLIENT} now ...")
            (clientSocket, _) = self.forwardSocket.accept()
            self.clientSocket = clientSocket
            iface0 = tcp_interface.TCPInterface(hostname="localhost", portNumber=self.nodes[0].TCPPort, connectNow=False)
            self.nodes[0].add_interface(iface0)
            iface0.myConnect()  # setup socket
            self.nodeThread = threading.Thread(target=self.node_reader, args=(), daemon=True)
            self.clientThread = threading.Thread(target=self.client_reader, args=(), daemon=True)
            self.nodeThread.start()
            self.clientThread.start()
            return iface0
        else:
            time.sleep(4)  # Allow instances to start up their TCP service

    def init_communication(self, iface0):
        try:
            print("[init_com] starting…")

            # NEW: serial mirror first (if enabled)
            self.init_serial_bridge()

            # TCP interfaces for non-mirror nodes
            for n in self.nodes[int(self.forwardToClient):]:
                if self.serialPort and n.nodeid == self.mirrorNode:
                    continue
                print(f"[init_com] setting up TCPInterface for node {n.nodeid} on port {n.TCPPort}")
                iface = tcp_interface.TCPInterface(hostname="localhost", portNumber=n.TCPPort)
                n.add_interface(iface)
            print("[init_com] all node interfaces added")

            if self.forwardToClient:
                self.clientConnected = True
                iface0.localNode.nodeNum = self.nodes[0].hwId
                iface0.connect()

            print("[init_com] setting configs for nodes…")
            for n in self.nodes:
                if n.iface is None:
                    continue
                requiresReboot = n.set_config()
                if requiresReboot and self.emulateCollisions and n.nodeid != len(self.nodes) - 1:
                    time.sleep(2)

            print("[init_com] calling reconnect_nodes() …")
            self.reconnect_nodes()

            pub.subscribe(self.on_receive, "meshtastic.receive.simulator")
            pub.subscribe(self.on_receive_metrics, "meshtastic.receive.telemetry")
            if self.forwardToClient:
                pub.subscribe(self.on_receive_all, "meshtastic.receive")

        except Exception as ex:
            print(f"[init_com] Error: Could not connect to native program: {ex}")
            self.close_nodes()
            sys.exit(1)

    # compatibility wrapper
    def reconnect_nodes(self):
        """Reconnect TCP-backed simulated nodes after config/reboot."""
        print("[reconnect] starting...")
        time.sleep(3)

        # close only TCP virtual nodes (skip serial mirror)
        for n in self.nodes[int(self.forwardToClient):]:
            if self.serialPort and n.nodeid == self.mirrorNode:
                continue
            try:
                if n.iface is not None:
                    n.iface.close()
            except Exception:
                pass
            n.iface = None

        time.sleep(2)

        # reopen TCP interfaces
        for n in self.nodes[int(self.forwardToClient):]:
            if self.serialPort and n.nodeid == self.mirrorNode:
                continue

            while n.iface is None:
                try:
                    iface = tcp_interface.TCPInterface(hostname="localhost", portNumber=n.TCPPort)
                    n.add_interface(iface)
                except Exception:
                    time.sleep(1)

            if self.emulateCollisions and n.nodeid != self.nodes[-1].nodeid:
                time.sleep(2)

        print("[reconnect] done.")

    # optional backwards-compat alias
    def reconnectNodes(self):
        return self.reconnect_nodes()

    # NEW
    def _dict_to_meshpacket(self, packet):
        mp = mesh_pb2.MeshPacket()
        mp.to = packet["to"]
        setattr(mp, "from", packet["from"])
        mp.id = packet["id"]
        mp.relay_node = packet.get("relayNode", mp.relay_node)
        mp.next_hop = packet.get("nextHop", mp.next_hop)
        mp.want_ack = packet.get("wantAck", mp.want_ack)
        mp.hop_limit = packet.get("hopLimit", mp.hop_limit)
        mp.hop_start = packet.get("hopStart", mp.hop_start)
        mp.via_mqtt = packet.get("viaMQTT", mp.via_mqtt)
        mp.channel = int(packet.get("channel", mp.channel))

        decoded = packet.get("decoded")
        if decoded:
            payload = decoded.get("payload", b"")
            if getattr(payload, "SerializeToString", None):
                payload = payload.SerializeToString()
            mp.decoded.payload = payload
            portnum = decoded.get("portnum", mp.decoded.portnum)
            if isinstance(portnum, str):
                try:
                    portnum = getattr(portnums_pb2.PortNum, portnum)
                except Exception:
                    pass
            mp.decoded.portnum = portnum
            mp.decoded.request_id = decoded.get("requestId", mp.decoded.request_id)
            mp.decoded.want_response = decoded.get("wantResponse", mp.decoded.want_response)
        else:
            cipher = packet.get("encrypted", b"")
            if getattr(cipher, "SerializeToString", None):
                cipher = cipher.SerializeToString()
            mp.encrypted = cipher
            mp.pki_encrypted = True
        return mp

    # NEW
    def _forward_to_serial(self, packet):
        if self.serial_iface is None:
            return
        try:
            tr = mesh_pb2.ToRadio()
            tr.packet.CopyFrom(self._dict_to_meshpacket(packet))
            self.serial_iface._sendToRadio(tr)
        except Exception as ex:
            print(f"[serial] forward failed: {ex}")

    # NEW: physical -> simulator
    def on_receive_serial(self, interface, packet):
        if self.serial_iface is None or interface is not self.serial_iface:
            return

        self._gc_seen()
        pkt_id = packet.get("id")
        if self._seen_recent(self._seen_to_serial, pkt_id):
            return
        self._mark_seen(self._seen_from_serial, pkt_id)

        tx = self.nodes[self.mirrorNode]
        receivers = [n for n in self.nodes if n.nodeid != tx.nodeid]
        rxs, rssis, snrs = self.calc_receivers(tx, receivers)
        self.forward_packet(rxs, packet, rssis, snrs)

    def forward_packet(self, receivers, packet, rssis, snrs):
        meshPacket = mesh_pb2.MeshPacket()

        # Common header fields
        meshPacket.to = packet["to"]
        setattr(meshPacket, "from", packet["from"])
        meshPacket.id = packet["id"]
        meshPacket.relay_node = packet.get("relayNode", meshPacket.relay_node)
        meshPacket.next_hop = packet.get("nextHop", meshPacket.next_hop)
        meshPacket.want_ack = packet.get("wantAck", meshPacket.want_ack)
        meshPacket.hop_limit = packet.get("hopLimit", meshPacket.hop_limit)
        meshPacket.hop_start = packet.get("hopStart", meshPacket.hop_start)
        meshPacket.via_mqtt = packet.get("viaMQTT", meshPacket.via_mqtt)
        meshPacket.channel = int(packet.get("channel", meshPacket.channel))

        data = packet["decoded"]["payload"]
        if getattr(data, "SerializeToString", None):
            data = data.SerializeToString()

        # If SimRadio wrapped plaintext, keep old behavior
        sim = packet["decoded"].get("simulator", {})
        sim_port = sim.get("portnum")

        if sim_port != "UNKNOWN_APP":
            # Normal path: forward SIMULATOR_APP wrapper
            if len(data) > mesh_pb2.Constants.DATA_PAYLOAD_LEN:
                raise Exception("Data payload too big")

            meshPacket.decoded.payload = data
            meshPacket.decoded.portnum = portnums_pb2.SIMULATOR_APP
            meshPacket.decoded.request_id = packet["decoded"].get("requestId", meshPacket.decoded.request_id)
            meshPacket.decoded.want_response = packet["decoded"].get("wantResponse", meshPacket.decoded.want_response)

        else:
            # Ciphertext path: UNKNOWN_APP => data is ciphertext bytes (no 0xEE flag)
            cipher = data
            if not cipher:
                return

            meshPacket.encrypted = cipher       # sets the oneof correctly
            meshPacket.pki_encrypted = True     # optional hint

        for i, rx in enumerate(receivers):
            meshPacket.rx_rssi = int(rssis[i])
            meshPacket.rx_snr = snrs[i]
            toRadio = mesh_pb2.ToRadio()
            toRadio.packet.CopyFrom(meshPacket)
            try:
                rx.iface._sendToRadio(toRadio)
            except Exception as ex:
                print(f"Error sending packet to radio!! ({ex})")


    def copy_packet(self, packet):
        time.sleep(0.01)
        try:
            # ORIGINAL: drop packets that already have simulator dict injected
            if 'simulator' in packet:
                return None

            # ORIGINAL: drop simulator wrapper packets
            if packet.get("decoded", {}).get("portnum") == "SIMULATOR_APP":
                return None

            # --- NEW: handle ciphertext packets (not decoded) ---
            # Depending on meshtastic python dict shape, encrypted bytes may appear as:
            #   packet["encrypted"]  (bytes)
            # or packet may have no "decoded" at all.
            if "decoded" not in packet or packet.get("decoded") is None:
                cipher = packet.get("encrypted", b"")
                if getattr(cipher, "SerializeToString", None):
                    cipher = cipher.SerializeToString()
                if not cipher:
                    return None

                mp = mesh_pb2.MeshPacket()
                mp.to = packet["to"]
                setattr(mp, "from", packet["from"])
                mp.id = packet["id"]
                mp.relay_node = packet.get("relayNode", mp.relay_node)
                mp.next_hop = packet.get("nextHop", mp.next_hop)
                mp.want_ack = packet.get("wantAck", mp.want_ack)
                mp.hop_limit = packet.get("hopLimit", mp.hop_limit)
                mp.hop_start = packet.get("hopStart", mp.hop_start)
                mp.via_mqtt = packet.get("viaMQTT", mp.via_mqtt)
                mp.channel = int(packet.get("channel", mp.channel))

                mp.encrypted = cipher
                mp.pki_encrypted = True  # optional hint (safe)

                fr = mesh_pb2.FromRadio()
                fr.packet.CopyFrom(mp)
                return fr

            # --- ORIGINAL decoded path (unchanged) ---
            data = packet["decoded"]["payload"]
            if getattr(data, "SerializeToString", None):
                data = data.SerializeToString()

            meshPacket = self.packet_from_packet(packet, data, packet["decoded"]["portnum"])
            fromRadio = mesh_pb2.FromRadio()
            fromRadio.packet.CopyFrom(meshPacket)
            return fromRadio

        except Exception:
            return None



    def showNodes(self, id=None):
        if id is not None:
            print('NodeDB as seen by node', id)
            self.nodes[id].iface.showNodes()
        else:
            for n in self.nodes:
                print('NodeDB as seen by node', n.nodeid)
                n.iface.showNodes()

    def send_broadcast(self, text, fromNode):
        self.get_node_iface_by_id(fromNode).sendText(text)

    def send_dm(self, text, fromNode, toNode):
        self.get_node_iface_by_id(fromNode).sendText(text, destinationId=self.node_id_to_hw_id(toNode), wantAck=True)

    def send_ping(self, fromNode, toNode):
        payload = str.encode("test string")
        self.get_node_iface_by_id(fromNode).sendData(
            payload, destinationId=self.node_id_to_hw_id(toNode),
            portNum=portnums_pb2.PortNum.REPLY_APP,
            wantAck=True, wantResponse=True
        )

    def trace_route(self, fromNode, toNode):
        r = mesh_pb2.RouteDiscovery()
        self.get_node_iface_by_id(fromNode).sendData(r, destinationId=self.node_id_to_hw_id(toNode), portNum=portnums_pb2.PortNum.TRACEROUTE_APP, wantResponse=True)

    def request_position(self, fromNode, toNode):
        self.get_node_iface_by_id(fromNode).sendPosition(destinationId=self.node_id_to_hw_id(toNode), wantResponse=True)

    def request_local_stats(self, toNode):
        r = telemetry_pb2.Telemetry()
        r.local_stats.CopyFrom(telemetry_pb2.LocalStats())
        self.get_node_iface_by_id(toNode).sendData(r, destinationId=self.node_id_to_hw_id(toNode), portNum=portnums_pb2.PortNum.TELEMETRY_APP, wantResponse=True)

    def get_node_iface_by_id(self, id):
        for n in self.nodes:
            if n.hwId == self.node_id_to_hw_id(id):
                return n.iface
        return None

    def node_id_to_dest(self, id):
        val = hex(self.node_id_to_hw_id(id)).strip('0x')
        return '!'+'0'*(8-len(val))+val

    def node_id_to_hw_id(self, id):
        return int(id) + HW_ID_OFFSET

    def send_from_to(self, fromNode, toNode):
        return self.get_node_iface_by_id(fromNode).getNode(self.node_id_to_dest(toNode))

    def on_receive(self, interface, packet):
        decoded = packet.get("decoded", {})

        if "requestId" in decoded:
            existingMsgId = next((m.localId for m in self.messages if m.packet["id"] == decoded["requestId"]), None)
            mId = existingMsgId if existingMsgId is not None else self.messageId + 1
            if existingMsgId is None:
                self.messageId += 1
        else:
            existingMsgId = next((m.localId for m in self.messages if m.packet["id"] == packet["id"]), None)
            if existingMsgId is not None:
                mId = existingMsgId
            else:
                self.messageId += 1
                mId = self.messageId

        rP = InteractivePacket(packet, mId)
        self.messages.append(rP)

        transmitter = next((n for n in self.nodes if n.hwId == packet.get("from")), None)
        if transmitter is None and hasattr(interface, "portNumber"):
            transmitter = next((n for n in self.nodes if n.TCPPort == interface.portNumber), None)

        if transmitter is not None:
            receivers = [n for n in self.nodes if n.nodeid != transmitter.nodeid]
            rxs, rssis, snrs = self.calc_receivers(transmitter, receivers)
            rP.setTxRxs(transmitter, rxs)
            rP.setRSSISNR(rssis, snrs)
            self.forward_packet(rxs, packet, rssis, snrs)
            self.graph.packets.append(rP)

            # NEW: mirror node simulator -> physical
            pkt_id = packet.get("id")
            self._gc_seen()
            if self.serial_iface is not None and transmitter.nodeid == self.mirrorNode:
                if not self._seen_recent(self._seen_from_serial, pkt_id):
                    self._mark_seen(self._seen_to_serial, pkt_id)
                    self._forward_to_serial(packet)

    def on_receive_metrics(self, interface, packet):
        fromNode = next((n for n in self.nodes if n.hwId == packet["from"]), None)
        if fromNode is not None:
            data = packet["decoded"]["payload"]
            if getattr(data, "SerializeToString", None):
                data = data.SerializeToString()
            telemetryPacket = telemetry_pb2.Telemetry()
            telemetryPacket.ParseFromString(data)
            telemetryDict = proto.MessageToDict(telemetryPacket)
            if 'deviceMetrics' in telemetryDict:
                deviceMetrics = telemetryDict['deviceMetrics']
                if 'time' in telemetryDict:
                    timestamp = int(telemetryDict['time'])
                    # Check whether it is not a duplicate
                    if len(fromNode.timestamps) == 0 or timestamp > fromNode.timestamps[-1]:
                        fromNode.timestamps.append(timestamp)
                        fromNode.channelUtilization.append(float(deviceMetrics.get('channelUtilization', 0)))
                        fromNode.airUtilTx.append(float(deviceMetrics.get('airUtilTx', 0)))
            elif 'localStats' in telemetryDict:
                localStats = telemetryDict['localStats']
                fromNode.numPacketsTx = localStats.get('numPacketsTx', fromNode.numPacketsTx)
                fromNode.numPacketsRx = localStats.get('numPacketsRx', fromNode.numPacketsRx)
                fromNode.numPacketsRxBad = localStats.get('numPacketsRxBad', fromNode.numPacketsRxBad)
                fromNode.numRxDupe = localStats.get('numRxDupe', fromNode.numRxDupe)
                fromNode.numTxRelay = localStats.get('numTxRelay', fromNode.numTxRelay)
                fromNode.numTxRelayCanceled = localStats.get('numTxRelayCanceled', fromNode.numTxRelayCanceled)

    def on_receive_all(self, interface, packet):
        if interface.portNumber == 4403:
            fromRadio = self.copy_packet(packet)
            if fromRadio is not None:
                # print("Forward", packet["decoded"])
                b = fromRadio.SerializeToString()
                bufLen = len(b)
                # We convert into a string, because the TCP code doesn't work with byte arrays
                header = bytes([0x94, 0xC3, (bufLen >> 8) & 0xFF, bufLen & 0xFF])
                self.clientSocket.send(header + b)

    def node_reader(self):
        while not self.wantExit and self.nodes[0].iface is not None:
            if self.clientConnected:
                break
            else:
                bytes = self.nodes[0].iface._readBytes(MAX_TO_FROM_RADIO_SIZE)
                if len(bytes) > 0:
                    # print(bytes)
                    self.clientSocket.send(bytes)

    def client_reader(self):
        while not self.wantExit:
            if self.nodes[0].iface is not None:
                bytes = self.clientSocket.recv(MAX_TO_FROM_RADIO_SIZE)
                if len(bytes) > 0:
                    self.nodes[0].iface._writeBytes(bytes)
            else:
                time.sleep(0.1)

    def calc_receivers(self, tx, receivers):
        rxs = []
        rssis = []
        snrs = []
        for rx in receivers:
            dist_3d = calc_dist(tx.x, rx.x, tx.y, rx.y, tx.z, rx.z)
            pathLoss = phy.estimate_path_loss(conf, dist_3d, conf.FREQ, tx.z, rx.z)
            RSSI = conf.PTX + tx.antennaGain + rx.antennaGain - pathLoss
            SNR = RSSI-conf.NOISE_LEVEL
            if RSSI >= conf.SENSMODEM[conf.MODEM]:
                rxs.append(rx)
                rssis.append(RSSI)
                snrs.append(SNR)
        return rxs, rssis, snrs

    def close_nodes(self):
        print("\nClosing all nodes...")
        pub.unsubAll()

        for n in self.nodes:
            try:
                if n.iface is not None:
                    n.iface.localNode.exitSimulator()
            except Exception:
                pass
            try:
                if n.iface is not None:
                    n.iface.close()
            except Exception:
                pass

        if self.serial_iface is not None:
            try:
                self.serial_iface.close()
            except Exception:
                pass

        # stop Docker container, ignore if already auto-removed
        if self.docker and getattr(self, "container", None) is not None:
            try:
                self.container.stop()
            except docker.errors.NotFound:
                # container already removed (auto_remove=True)
                pass
            except Exception as e:
                print(f"Warning: could not stop container cleanly: {e}")

        # correctly set the exit flag for reader threads
        if self.forwardToClient:
            self.wantExit = True      # <- was self._wantExit
            try:
                self.forwardSocket.close()
            except Exception:
                pass
            try:
                self.clientSocket.close()
            except Exception:
                pass


class CommandProcessor(cmd.Cmd):

    def cmdloop(self, sim):
        self.sim = sim
        print("Type 'help' to list the available commands for sending messages. Type 'plot' to show the routes or 'exit' to exit the simulator.")
        return cmd.Cmd.cmdloop(self)

    def do_broadcast(self, line):
        """broadcast <fromNode> <txt>
        Send a broadcast from node \x1B[3mfromNode\x1B[0m with text \x1B[3mtxt\x1B[0m."""
        arguments = line.split()
        if len(arguments) < 2:
            print('Please use the syntax: "broadcast <fromNode> <txt>"')
            return False
        fromNode = int(arguments[0])
        if self.sim.get_node_iface_by_id(fromNode) is None:
            print(f'Node ID {fromNode} is not in the list of nodes.')
            return False
        txt = " ".join(arguments[1:])
        print(f'Instructing node {fromNode} to broadcast "{txt}" (message ID = {self.sim.messageId+1})')
        self.sim.send_broadcast(txt, fromNode)

    def do_dm(self, line):
        """DM <fromNode> <toNode> <txt>
        Send a Direct Message from node \x1B[3mfromNode\x1B[0m to node \x1B[3mtoNode\x1B[0m with text \x1B[3mtxt\x1B[0m."""
        arguments = line.split()
        if len(arguments) < 3:
            print('Please use the syntax: "DM <fromNode> <toNode> <txt>"')
            return False
        fromNode = int(arguments[0])
        if self.sim.get_node_iface_by_id(fromNode) is None:
            print(f'Node ID {fromNode} is not in the list of nodes.')
            return False
        toNode = int(arguments[1])
        if self.sim.get_node_iface_by_id(toNode) is None:
            print(f'Node ID {toNode} is not in the list of nodes.')
            return False
        txt = " ".join(arguments[2:])
        print(f'Instructing node {fromNode} to DM node {toNode} "{txt}" (message ID = {self.sim.messageId+1})')
        self.sim.send_dm(txt, fromNode, toNode)

    def do_ping(self, line):
        """ping <fromNode> <toNode>
        Send ping from node \x1B[3mfromNode\x1B[0m to node \x1B[3mtoNode\x1B[0m."""
        arguments = line.split()
        if len(arguments) != 2:
            print('Please use the syntax: "ping <fromNode> <toNode>"')
            return False
        fromNode, toNode = map(int, arguments)
        if self.sim.get_node_iface_by_id(fromNode) is None:
            print('Node ID', fromNode, 'is not in the list of nodes.')
            return False
        if self.sim.get_node_iface_by_id(toNode) is None:
            print('Node ID', toNode, 'is not in the list of nodes.')
            return False
        print(f'Instructing node {fromNode} to send ping to node {toNode} (message ID = {self.sim.messageId+1})')
        self.sim.send_ping(fromNode, toNode)

    def do_traceroute(self, line):
        """traceroute <fromNode> <toNode>
        Send a traceroute request from node \x1B[3mfromNode\x1B[0m to node \x1B[3mtoNode\x1B[0m."""
        arguments = line.split()
        if len(arguments) != 2:
            print('Please use the syntax: "traceroute <fromNode> <toNode>"')
            return False
        fromNode, toNode = map(int, arguments)
        if self.sim.get_node_iface_by_id(fromNode) is None:
            print('Node ID', fromNode, 'is not in the list of nodes.')
            return False
        if self.sim.get_node_iface_by_id(toNode) is None:
            print('Node ID', toNode, 'is not in the list of nodes.')
            return False
        print(f'Instructing node {fromNode} to send traceroute request to node {toNode} (message ID = {self.sim.messageId+1})')
        print(f'This takes a while, the result will be in the log of node {fromNode}.')
        self.sim.trace_route(fromNode, toNode)

    def do_req_pos(self, line):
        """reqPos <fromNode> <toNode>
        Send a position request from node \x1B[3mfromNode\x1B[0m to node \x1B[3mtoNode\x1B[0m."""
        arguments = line.split()
        if len(arguments) != 2:
            print('Please use the syntax: "reqPos <fromNode> <toNode>"')
            return False
        fromNode, toNode = map(int, arguments)
        if self.sim.get_node_iface_by_id(fromNode) is None:
            print(f'Node ID {fromNode} is not in the list of nodes.')
            return False
        if self.sim.get_node_iface_by_id(toNode) is None:
            print(f'Node ID {toNode} is not in the list of nodes.')
            return False
        print(f'Instructing node {fromNode} to send position request to node {toNode} (message ID = {self.sim.messageId+1})')
        self.sim.request_position(fromNode, toNode)

    def do_nodes(self, line):
        """nodes <id0> [id1, etc.]
        Show the node list as seen by node(s) \x1B[3mid0\x1B[0m, \x1B[3mid1\x1B[0m., etc."""
        arguments = line.split()
        if len(arguments) < 1:
            print('Please use the syntax: "nodes <id0> [id1, etc.]"')
            return False
        for n in arguments:
            if self.sim.get_node_iface_by_id(n) is None:
                print(f'Node ID {n} is not in the list of nodes.')
                continue
            self.sim.showNodes(int(n))

    def do_remove(self, line):
        """remove <id>
        Remove node \x1B[3mid\x1B[0m from the current simulation."""
        arguments = line.split()
        if len(arguments) < 1:
            print('Please use the syntax: "remove <id>"')
            return False
        nodeId = (int(arguments[0]))
        if self.sim.get_node_iface_by_id(nodeId) is None:
            print(f'Node ID {nodeId} is not in the list of nodes.')
        else:
            self.sim.get_node_iface_by_id(nodeId).localNode.exitSimulator()
            self.sim.get_node_iface_by_id(nodeId).close()
            del self.sim.nodes[nodeId]

    def do_plot(self, line):
        """plot
        Plot the routes of messages sent and airtime statistics."""
        if self.sim.emulateCollisions:
            for n in self.sim.nodes:
                self.sim.request_local_stats(n.nodeid)
            time.sleep(1)
        self.sim.graph.plot_metrics(self.sim.nodes)
        self.sim.graph.init_routes(self.sim)
        return True

    def do_exit(self, line):
        """exit
        Exit the simulator without plotting routes."""
        self.sim.close_nodes()
        return True
