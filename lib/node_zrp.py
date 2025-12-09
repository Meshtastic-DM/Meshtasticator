from lib.node import MeshNode
from lib.packet import NODENUM_BROADCAST, MeshMessage
from lib.packet_zrp import MeshPacket_ZRP
import simpy
from dataclasses import dataclass


@dataclass
class IARPEntry:
    destId: int
    nextHop: int
    distance: int       # hop count within zone
    seq_num: int        # versioning for freshness
    last_updated: float # env.now when last updated


class MeshNode_ZRP(MeshNode):
    """
    ZRP-enabled node.

    Currently:
      - IARP (intrazone proactive routing) with periodic updates.
      - IERP skeleton:
          * RREQs are bordercast to peripheral nodes (distance == zone_radius).
          * At each peripheral, packet is unwrapped and `ierp_destId` is used.
          * If hop_count > ZRP_ZONE_RADIUS, add an entry in IERP table.
          * RREQs are dropped when hop_count > 7 / hopLimit < 0.
          * RREQ carries `covered_nodes`:
              - Origin: adds all its intrazone nodes.
              - Each peripheral: adds its own intrazone nodes and
                forwards only to peripheral nodes NOT in `covered_nodes`.
      - Data / ACK handling uses MeshPacket_ZRP.
      - Unicast send:
          * If dest is in IARP table and within zone_radius → send via nextHop.
          * Else → (for now) best-effort flooding; IERP RREQ is not yet wired
            to pending data queues.
    """

    def __init__(
        self,
        conf,
        nodes,
        env,
        bc_pipe,
        nodeid,
        period,
        messages,
        packetsAtN,
        packets,
        delays,
        nodeConfig,
        messageSeq,
        verboseprint,
    ):
        super().__init__(
            conf,
            nodes,
            env,
            bc_pipe,
            nodeid,
            period,
            messages,
            packetsAtN,
            packets,
            delays,
            nodeConfig,
            messageSeq,
            verboseprint,
        )

        # Mark router type as ZRP
        self.conf.SELECTED_ROUTER_TYPE = self.conf.ROUTER_TYPE.ZRP

        # Resource for transmit, similar to AODV
        self.transmitter = simpy.Resource(env, capacity=1)

        # ------------- ZRP / IARP state ----------------
        self.zone_radius = getattr(self.conf, "ZRP_ZONE_RADIUS", 2)
        self.iarp_table: dict[int, IARPEntry] = {}
        self.iarp_seq_num = 0

        # IARP periodic update interval (ms)
        self.iarp_period_msec = getattr(self.conf, "IARP_PERIOD_MSEC", 1 * 60 * 1000)

        # --------- IERP skeleton state ----------
        self.ierp_seq_num = 0
        self.ierp_table: dict[int, IARPEntry] = {}  # coarse interzone routes
        self.pending_ierp = {}  # key: queryDestId, value: list of DATA packets
        self.processed_ierp_rreq = set()  # (origTxNodeId, ierp_id)
        self.processed_ierp_rrep = set()  # (origTxNodeId, ierp_id)

        # Start periodic IARP process
        self.env.process(self._iarp_periodic_process())

    # =====================================================================
    # Helpers
    # =====================================================================

    def get_peripheral_neighbors(self):
        """Nodes at exactly zone_radius hops in IARP table."""
        return [
            entry
            for entry in self.iarp_table.values()
            if entry.distance == self.zone_radius
        ]

    def get_intrazone_nodes_set(self):
        """
        All nodes in my zone (distance <= zone_radius) plus myself.
        Used to populate / grow covered_nodes for IERP RREQ.
        """
        s = {self.nodeid}
        for entry in self.iarp_table.values():
            if entry.distance <= self.zone_radius:
                s.add(entry.destId)
        return s

    # =====================================================================
    # ZRP send_packet: DATA (unicast + broadcast)
    # =====================================================================

    def send_packet(self, destId, data=None, wantAck=True, is_sdn_update=False):
        """
        ZRP-aware send_packet:

        - Always builds MeshPacket_ZRP.
        - Broadcast:
            destId == NODENUM_BROADCAST → flood as DATA broadcast.
        - Unicast:
            If IARP has dest and distance <= zone_radius:
                → send via nextHop (intra-zone routing).
            Else:
                → currently: best-effort directed flood (DATA + destId).
                  IERP route discovery is implemented but not yet wired
                  to data forwarding / pending queues.
        """

        plen = 20

        # Logical message record (same style as AODV)
        self.messageSeq["val"] += 1
        messageSeq = self.messageSeq["val"]
        self.messages.append(
            MeshMessage(self.nodeid, destId, self.env.now, messageSeq)
        )

        # Base DATA packet (ZRP)
        base_packet = MeshPacket_ZRP(
            self.conf,
            self.nodes,
            origTxNodeId=self.nodeid,
            destId=destId,
            txNodeId=self.nodeid,
            packetLen=plen,
            seq=messageSeq,
            genTime=self.env.now,
            wantAck=wantAck,
            isAck=False,
            requestId=None,
            txTime=self.env.now,
            verboseprint=self.verboseprint,
            packet_type=None,      # None → DATA (non-IARP, non-IERP-control)
            iarp_seq_num=None,
            ierp_type=None,
            ierp_id=None,
            hop_count=0,
        )
        base_packet.data = data
        base_packet.is_sdn_update = is_sdn_update

        # =======================
        # Broadcast DATA
        # =======================
        if destId == NODENUM_BROADCAST:
            pNew = base_packet
            default_hl = getattr(self.conf, "HOP_LIMIT", self.zone_radius)
            pNew.hopLimit = getattr(pNew, "hopLimit", default_hl)

            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "broadcasting ZRP DATA packet", pNew.seq,
            )
            self.packets.append(pNew)
            self.env.process(self.transmit(pNew))
            return base_packet

        # =======================
        # Unicast DATA
        # =======================

        entry = self.iarp_table.get(destId, None)

        # Intra-zone: we have an IARP route and dest is within zone radius
        if entry is not None and entry.distance <= self.zone_radius:
            pNew = base_packet
            pNew.next_hop = entry.nextHop
            default_hl = getattr(self.conf, "HOP_LIMIT", self.zone_radius)
            pNew.hopLimit = min(default_hl, self.zone_radius)

            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "sending ZRP unicast packet", pNew.seq,
                "to", destId,
                "via nextHop", entry.nextHop,
                "(distance", entry.distance, ")",
            )
            self.packets.append(pNew)
            self.env.process(self.transmit(pNew))
            return base_packet

        # Interzone or unknown: no IARP route available
        self.verboseprint(
            "At time", round(self.env.now, 3),
            "node", self.nodeid,
            "has no IARP route to", destId,
            "→ IERP would be triggered (RREQ skeleton implemented).",
        )

        # TODO (full ZRP): queue & start IERP
        # self.pending_ierp.setdefault(destId, []).append(base_packet)
        # self.initiate_route_discovery(destId)
        # return base_packet

        # For now: best-effort directed flood
        pNew = base_packet
        default_hl = getattr(self.conf, "HOP_LIMIT", self.zone_radius * 2)
        pNew.hopLimit = getattr(pNew, "hopLimit", default_hl)

        self.verboseprint(
            "At time", round(self.env.now, 3),
            "node", self.nodeid,
            "flooding ZRP DATA packet", pNew.seq,
            "towards dest", destId,
            "(IERP not fully wired yet)",
        )
        self.packets.append(pNew)
        self.env.process(self.transmit(pNew))
        return base_packet

    # =====================================================================
    # IARP: Intrazone proactive routing
    # =====================================================================

    def _iarp_periodic_process(self):
        """
        Periodically broadcast IARP updates within the zone.

        Every iarp_period_msec, this node sends an IARP packet
        advertising itself with distance 0 and hopLimit = ZRP_ZONE_RADIUS.
        """
        while True:
            nextGen = self.get_next_time(self.iarp_period_msec)
            if nextGen < 0:
                break
            yield self.env.timeout(nextGen)

            # Increment local IARP sequence number
            self.iarp_seq_num += 1

            # Logical message record
            self.messageSeq["val"] += 1
            messageSeq = self.messageSeq["val"]
            self.messages.append(
                MeshMessage(self.nodeid, NODENUM_BROADCAST, self.env.now, messageSeq)
            )

            # Build an IARP control packet
            p = MeshPacket_ZRP(
                self.conf,
                self.nodes,
                origTxNodeId=self.nodeid,
                destId=NODENUM_BROADCAST,   # zone-broadcast
                txNodeId=self.nodeid,
                packetLen=0,                 # overridden by MeshPacket_ZRP for IARP
                seq=messageSeq,
                genTime=self.env.now,
                wantAck=False,
                isAck=False,
                requestId=None,
                txTime=self.env.now,
                verboseprint=self.verboseprint,
                packet_type="IARP",
                iarp_seq_num=self.iarp_seq_num,
                hop_count=0,                 # origin: 0 hops from itself
            )
            p.hopLimit = self.zone_radius

            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "broadcasting IARP update with seq", self.iarp_seq_num,
                "hopLimit", p.hopLimit,
            )

            self.packets.append(p)
            self.env.process(self.transmit(p))

    def handle_iarp(self, packet: MeshPacket_ZRP):
        if not isinstance(packet, MeshPacket_ZRP) or packet.packet_type != "IARP":
            return

        src = packet.origTxNodeId

        # Do not install routes to yourself
        if src == self.nodeid:
            return

        distance = getattr(packet, "hop_count", 0) + 1
        seq_num = packet.iarp_seq_num
        now = self.env.now

        existing = self.iarp_table.get(src, None)

        if (
            existing is None
            or seq_num > existing.seq_num
            or (seq_num == existing.seq_num and distance < existing.distance)
        ):
            self.iarp_table[src] = IARPEntry(
                destId=src,
                nextHop=packet.txNodeId,
                distance=distance,
                seq_num=seq_num,
                last_updated=now,
            )
            self.verboseprint(
                "At time", round(now, 3),
                "node", self.nodeid,
                "updated IARP entry for", src,
                "via nextHop", packet.txNodeId,
                "distance", distance,
                "seq", seq_num,
            )

    def get_iarp_table(self):
        info = {}
        for destId, entry in self.iarp_table.items():
            info[destId] = {
                "nextHop": entry.nextHop,
                "distance": entry.distance,
                "seq_num": entry.seq_num,
                "last_updated": entry.last_updated,
            }
        return info

    # =====================================================================
    # IERP / BRP – skeleton (bordercast RREQ only, with covered_nodes)
    # =====================================================================

    def initiate_route_discovery(self, destId):
        """
        IERP RREQ bordercast skeleton.

        - destId: query destination (DM/app node).
        - Each IERP-RREQ copy:
            * destId      = peripheral node (distance == zone_radius).
            * ierp_destId = real query destination.
        - covered_nodes:
            * At origin: all nodes in my intrazone set (including myself).
        """
        self.ierp_seq_num += 1
        ierp_id = self.ierp_seq_num

        peripherals = self.get_peripheral_neighbors()
        if not peripherals:
            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "has no peripheral neighbors for IERP; discovery aborted.",
            )
            return

        # Build initial covered_nodes = my intrazone (including me)
        covered_nodes = list(self.get_intrazone_nodes_set())

        for entry in peripherals:
            rreq = MeshPacket_ZRP(
                self.conf,
                self.nodes,
                origTxNodeId=self.nodeid,
                destId=entry.destId,      # peripheral node
                txNodeId=self.nodeid,
                packetLen=0,
                seq=ierp_id,
                genTime=self.env.now,
                wantAck=False,
                isAck=False,
                requestId=None,
                txTime=self.env.now,
                verboseprint=self.verboseprint,
                packet_type="IERP",
                iarp_seq_num=None,
                ierp_type="RREQ",
                ierp_id=ierp_id,
                hop_count=0,
                ierp_destId=destId,       # query destination
                covered_nodes=covered_nodes,
            )
            rreq.hopLimit = 7
            rreq.next_hop = entry.nextHop

            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "bordercasting IERP RREQ", rreq.ierp_id,
                "for query dest", destId,
                "to peripheral", entry.destId,
                "via nextHop", entry.nextHop,
                "covered_nodes", rreq.covered_nodes,
            )
            self.packets.append(rreq)
            self.env.process(self.transmit(rreq))

    def handle_ierp_rreq(self, packet: MeshPacket_ZRP):
        if packet.packet_type != "IERP" or packet.ierp_type != "RREQ":
            return

        key = (packet.origTxNodeId, packet.ierp_id)
        if key in self.processed_ierp_rreq:
            self.verboseprint("ZRP: duplicate IERP RREQ dropped", key)
            return
        self.processed_ierp_rreq.add(key)

        # Unwrap: query destination is carried in ierp_destId
        query_dest = getattr(packet, "ierp_destId", None)
        if query_dest is None:
            self.verboseprint("ZRP: IERP RREQ without ierp_destId dropped")
            return

        # -------------------------------
        # Update hop_count and hopLimit
        # -------------------------------
        packet.hop_count = getattr(packet, "hop_count", 0) + 1
        if packet.hopLimit is not None:
            packet.hopLimit -= 1
            if packet.hopLimit < 0:
                self.verboseprint(
                    "At time", round(self.env.now, 3),
                    "node", self.nodeid,
                    "drops IERP RREQ", packet.ierp_id,
                    "due to hopLimit",
                )
                return

        # Drop if hop_count exceeds 7 (global bound)
        if packet.hop_count > 7:
            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "drops IERP RREQ", packet.ierp_id,
                "because hop_count", packet.hop_count, "> 7",
            )
            return

        # -------------------------------
        # Grow covered_nodes at this node
        # -------------------------------
        covered = set(getattr(packet, "covered_nodes", []) or [])
        covered |= self.get_intrazone_nodes_set()   # add my intrazone nodes
        packet.covered_nodes = list(covered)

        # If I am the query destination, you'd generate RREP here
        if self.nodeid == query_dest:
            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "is query dest", query_dest,
                "→ would send IERP RREP (not yet implemented).",
            )
            # TODO: call send_ierp_rrep(packet)
            return

        # If IARP has route to query_dest, you could answer with RREP
        iarp_entry = self.iarp_table.get(query_dest)
        if iarp_entry and iarp_entry.distance <= self.zone_radius:
            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "has IARP route to query dest", query_dest,
                "→ would send IERP RREP (not yet implemented).",
            )
            # TODO: call send_ierp_rrep(packet)
            return

        # If hop_count > ZRP_ZONE_RADIUS, store coarse interzone info
        if packet.hop_count > self.zone_radius:
            origin = packet.origTxNodeId
            self.ierp_table[origin] = IARPEntry(
                destId=origin,
                nextHop=packet.txNodeId,
                distance=packet.hop_count,
                seq_num=0,
                last_updated=self.env.now,
            )

        # Bordercast further to my peripherals, but only to nodes
        # whose destId is NOT in covered_nodes
        peripherals = self.get_peripheral_neighbors()
        if not peripherals:
            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "no peripherals to continue IERP RREQ", packet.ierp_id,
            )
            return

        for entry in peripherals:
            # avoid immediate backtracking through same nextHop
            if entry.nextHop == packet.txNodeId:
                continue

            # skip peripherals that are already covered
            if entry.destId in covered:
                continue

            fwd = MeshPacket_ZRP(
                self.conf,
                self.nodes,
                origTxNodeId=packet.origTxNodeId,
                destId=entry.destId,     # new peripheral
                txNodeId=self.nodeid,
                packetLen=packet.packetLen,
                seq=packet.seq,
                genTime=packet.genTime,
                wantAck=False,
                isAck=False,
                requestId=None,
                txTime=self.env.now,
                verboseprint=self.verboseprint,
                packet_type="IERP",
                iarp_seq_num=None,
                ierp_type="RREQ",
                ierp_id=packet.ierp_id,
                hop_count=packet.hop_count,
                ierp_destId=query_dest,
                covered_nodes=list(covered),
            )
            fwd.hopLimit = packet.hopLimit
            fwd.next_hop = entry.nextHop

            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "bordercasts IERP RREQ", fwd.ierp_id,
                "for query dest", query_dest,
                "to peripheral", entry.destId,
                "via nextHop", entry.nextHop,
                "hop_count", fwd.hop_count,
                "covered_nodes", fwd.covered_nodes,
            )
            self.packets.append(fwd)
            self.env.process(self.transmit(fwd))

    def handle_ierp_rrep(self, packet):
        # Stub – to be implemented when you wire RREP back to origin
        pass

    def handle_zrp_control(self, packet):
        if isinstance(packet, MeshPacket_ZRP):
            if packet.packet_type == "IARP":
                self.handle_iarp(packet)
            elif packet.packet_type == "IERP":
                if packet.ierp_type == "RREQ":
                    self.handle_ierp_rreq(packet)
                elif packet.ierp_type == "RREP":
                    self.handle_ierp_rrep(packet)

    # =====================================================================
    # RECEIVE: IARP (no flooding) + IERP + DATA/ACK
    # =====================================================================

    def receive(self, pipe):
        while True:
            packet = yield pipe.get()

            # ----------------- Start of reception -----------------
            if (
                packet.sensedByN[self.nodeid]
                and not packet.collidedAtN[self.nodeid]
                and packet.onAirToN[self.nodeid]
            ):
                if not self.isTransmitting:
                    self.verboseprint(
                        "At time", round(self.env.now, 3),
                        "node", self.nodeid,
                        "started receiving packet", packet.seq,
                        "from", packet.txNodeId,
                    )
                    packet.onAirToN[self.nodeid] = False
                    self.isReceiving.append(True)
                else:
                    self.verboseprint(
                        "At time", round(self.env.now, 3),
                        "node", self.nodeid,
                        "was transmitting, so could not receive packet", packet.seq,
                    )
                    packet.sensedByN[self.nodeid] = False
                    packet.onAirToN[self.nodeid] = False

            # ----------------- End of reception -----------------
            elif packet.sensedByN[self.nodeid]:
                try:
                    self.isReceiving[self.isReceiving.index(True)] = False
                except Exception:
                    pass

                self.airUtilization += packet.timeOnAir

                if packet.collidedAtN[self.nodeid]:
                    self.verboseprint(
                        "At time", round(self.env.now, 3),
                        "node", self.nodeid,
                        "could not decode packet.",
                    )
                    continue

                packet.receivedAtN[self.nodeid] = True
                self.verboseprint(
                    "At time", round(self.env.now, 3),
                    "node", self.nodeid,
                    "received packet", packet.seq,
                    "with delay", round(self.env.now - packet.genTime, 2),
                )
                self.delays.append(self.env.now - packet.genTime)

                # ==================================================
                # IARP control packets – NO managed flooding now
                # ==================================================
                if isinstance(packet, MeshPacket_ZRP) and packet.packet_type == "IARP":
                    self.handle_iarp(packet)
                    continue

                # ==================================================
                # IERP control packets – RREQ / RREP
                # ==================================================
                if isinstance(packet, MeshPacket_ZRP) and packet.packet_type == "IERP":
                    if packet.ierp_type == "RREQ":
                        self.handle_ierp_rreq(packet)
                    elif packet.ierp_type == "RREP":
                        self.handle_ierp_rrep(packet)
                    continue

                # ==================================================
                # Data / ACK handling (non-IARP / non-IERP-control)
                # ==================================================
                if packet.destId == self.nodeid or packet.destId == NODENUM_BROADCAST:
                    # Generate ACK if required
                    if not packet.isAck and packet.wantAck:
                        self.messageSeq["val"] += 1
                        messageSeq = self.messageSeq["val"]
                        self.messages.append(
                            MeshMessage(
                                self.nodeid,
                                packet.origTxNodeId,
                                self.env.now,
                                messageSeq,
                            )
                        )
                        ack_packet = MeshPacket_ZRP(
                            self.conf,
                            self.nodes,
                            self.nodeid,             # origTxNodeId
                            packet.origTxNodeId,     # destId
                            self.nodeid,             # txNodeId
                            10,                      # packetLen
                            messageSeq,
                            self.env.now,
                            False,                   # wantAck
                            True,                    # isAck
                            packet.seq,              # requestId
                            self.env.now,
                            self.verboseprint,
                            None,                    # packet_type = None (DATA/ACK)
                            None,                    # iarp_seq_num
                            None,                    # ierp_type
                            None,                    # ierp_id
                            getattr(packet, "hop_count", 0) + 1,
                        )
                        self.packets.append(ack_packet)
                        self.env.process(self.transmit(ack_packet))
                        self.verboseprint(
                            "At time", round(self.env.now, 3),
                            "node", self.nodeid,
                            "sent ACK for packet", packet.seq,
                            "to", packet.origTxNodeId,
                        )

                    self.verboseprint(
                        "At time", round(self.env.now, 3),
                        "node", self.nodeid,
                        "received packet", packet.seq,
                        "from", packet.origTxNodeId,
                    )

                    if not packet.isAck:
                        orginTxNodeId = packet.origTxNodeId
                        orginTxNode = None
                        for n in self.nodes:
                            if n.nodeid == orginTxNodeId:
                                orginTxNode = n
                                break

                        # Sensor → Control stats
                        if orginTxNode and orginTxNode.simRole == "Sensor":
                            self.verboseprint(
                                "At time", round(self.env.now, 3),
                                "node", self.nodeid,
                                "is a Control node receiving a packet from Sensor node",
                                orginTxNodeId,
                            )
                            if packet.seq not in self.SensorPacketsReceived:
                                self.SensorPacketsReceived[packet.seq] = 0
                                if (
                                    packet.origTxNodeId
                                    not in self.SensorPacketsReceivedOrigId
                                ):
                                    self.SensorPacketsReceivedOrigId[
                                        packet.origTxNodeId
                                    ] = {}
                                self.SensorPacketsReceivedOrigId[packet.origTxNodeId][
                                    packet.seq
                                ] = 0
                                if (
                                    packet.origTxNodeId
                                    not in self.SensorPacketsDelays
                                ):
                                    self.SensorPacketsDelays[packet.origTxNodeId] = []
                                self.SensorPacketsDelays[packet.origTxNodeId].append(
                                    self.env.now - packet.genTime
                                )
                            self.SensorPacketsReceived[packet.seq] += 1
                            self.SensorPacketsReceivedOrigId[packet.origTxNodeId][
                                packet.seq
                            ] += 1

                        # Control_Center broadcast handling
                        elif orginTxNode and orginTxNode.simRole == "Control_Center":
                            if packet.seq not in self.BroadcastPacketsReceived:
                                self.BroadcastPacketsReceived[packet.seq] = 0
                                if not self.isClientMute:
                                    self.verboseprint(
                                        "At time", round(self.env.now, 3),
                                        "node", self.nodeid,
                                        "rebroadcasts received broadcast packet",
                                        packet.seq,
                                    )
                                    pNew = MeshPacket_ZRP(
                                        self.conf,
                                        self.nodes,
                                        packet.origTxNodeId,
                                        packet.destId,
                                        self.nodeid,
                                        packet.packetLen,
                                        packet.seq,
                                        packet.genTime,
                                        packet.wantAck,
                                        packet.isAck,
                                        None,
                                        self.env.now,
                                        self.verboseprint,
                                        None,   # packet_type = None (DATA)
                                        None,
                                        None,
                                        None,
                                        getattr(packet, "hop_count", 0) + 1,
                                    )
                                    self.packets.append(pNew)
                                    self.env.process(self.transmit(pNew))
                                    self.verboseprint(
                                        "At time", round(self.env.now, 3),
                                        "node", self.nodeid,
                                        "rebroadcasted broadcast packet", pNew.seq,
                                    )
                                    if (
                                        packet.origTxNodeId
                                        not in self.BroadcastPacketsDelays
                                    ):
                                        self.BroadcastPacketsDelays[
                                            packet.origTxNodeId
                                        ] = []
                                    self.BroadcastPacketsDelays[
                                        packet.origTxNodeId
                                    ].append(self.env.now - packet.genTime)
                            self.BroadcastPacketsReceived[packet.seq] += 1

                        # DM unicast stats
                        elif orginTxNode and orginTxNode.simRole == "DM":
                            if packet.seq not in self.DMPacketsReceived:
                                self.DMPacketsReceived[packet.seq] = 0
                                if (
                                    packet.origTxNodeId
                                    not in self.DMPacketsReceivedOrigId
                                ):
                                    self.DMPacketsReceivedOrigId[
                                        packet.origTxNodeId
                                    ] = {}
                                self.DMPacketsReceivedOrigId[packet.origTxNodeId][
                                    packet.seq
                                ] = 0
                                if (
                                    packet.origTxNodeId
                                    not in self.DMPacketsDelays
                                ):
                                    self.DMPacketsDelays[packet.origTxNodeId] = []
                                self.DMPacketsDelays[packet.origTxNodeId].append(
                                    self.env.now - packet.genTime
                                )
                            self.DMPacketsReceived[packet.seq] += 1
                            self.DMPacketsReceivedOrigId[packet.origTxNodeId][
                                packet.seq
                            ] += 1

                    else:
                        # ACK receive stats
                        if self.simRole == "Sensor":
                            if packet.seq not in self.SensorPacketsAcked:
                                self.SensorPacketsAcked[packet.seq] = 0
                                self.ACKPacketsDelays.append(
                                    self.env.now - packet.genTime
                                )
                            self.SensorPacketsAcked[packet.seq] += 1
                        elif self.simRole == "DM":
                            if packet.seq not in self.DMPacketsAcked:
                                self.DMPacketsAcked[packet.seq] = 0
                                self.ACKPacketsDelays.append(
                                    self.env.now - packet.genTime
                                )
                            self.DMPacketsAcked[packet.seq] += 1

                else:
                    # Not for me and not pure broadcast → optional forwarding (data)
                    if not self.isClientMute:
                        self.verboseprint(
                            "At time", round(self.env.now, 3),
                            "node", self.nodeid,
                            "rebroadcasts received packet", packet.seq,
                        )
                        pNew = MeshPacket_ZRP(
                            self.conf,
                            self.nodes,
                            packet.origTxNodeId,
                            packet.destId,
                            self.nodeid,
                            packet.packetLen,
                            packet.seq,
                            packet.genTime,
                            packet.wantAck,
                            packet.isAck,
                            None,
                            self.env.now,
                            self.verboseprint,
                            None,   # packet_type = None (DATA)
                            None,
                            None,
                            None,
                            getattr(packet, "hop_count", 0) + 1,
                        )
                        self.packets.append(pNew)
                        self.env.process(self.transmit(pNew))
                        self.verboseprint(
                            "At time", round(self.env.now, 3),
                            "node", self.nodeid,
                            "rebroadcasted packet", pNew.seq,
                            "to", pNew.destId,
                        )
                    else:
                        self.verboseprint(
                            "At time", round(self.env.now, 3),
                            "node", self.nodeid,
                            "dropped packet", packet.seq,
                            "because client is muted",
                        )

                # ----------------- ACK bookkeeping for queue -----------------
                for sentPacket in self.packets:
                    if sentPacket.txNodeId == self.nodeid and sentPacket.seq == packet.seq:
                        self.verboseprint(
                            "At time", round(self.env.now, 3),
                            "node", self.nodeid,
                            "received implicit ACK for message in queue.",
                        )
                        sentPacket.ackReceived = True
                    if (
                        sentPacket.origTxNodeId == self.nodeid
                        and packet.isAck
                        and sentPacket.seq == packet.requestId
                    ):
                        self.verboseprint(
                            "At time", round(self.env.now, 3),
                            "node", self.nodeid,
                            "received real ACK.",
                        )
                        sentPacket.ackReceived = True
