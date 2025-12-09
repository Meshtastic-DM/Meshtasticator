from lib.node import MeshNode
from lib.packet import NODENUM_BROADCAST, MeshMessage
from lib.packet_zrp import MeshPacket_ZRP
import simpy
import random
import math
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
      - Implements ONLY IARP (intrazone proactive routing).
      - Starts a periodic IARP update process.
      - Uses managed flooding for IARP within ZRP_ZONE_RADIUS hops.
      - Data / ACK handling uses MeshPacket_ZRP.
      - Unicast send:
          * If dest is in IARP table and within zone_radius → send via nextHop.
          * Else → placeholder for IERP (currently: best-effort flooding).
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

        # Placeholder for future IERP/BRP usage
        self.pending_ierp = {}  # key: destId, value: list of packets waiting for route

        # Start periodic IARP process
        self.env.process(self._iarp_periodic_process())

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
                → placeholder for IERP: currently best-effort flooding.
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
            packet_type=None,      # None → DATA (non-IARP)
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
            # hopLimit for broadcast: use config if present, else at least zone_radius
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
            # Keep hops constrained to the zone
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
            "→ would trigger IERP (placeholder).",
        )

        # TODO: proper IERP/BRP:
        #   - enqueue base_packet in self.pending_ierp[destId]
        #   - send ZRP IERP-RREQ etc.
        # For now: best-effort directed flood (DATA with destId, no next_hop)

        pNew = base_packet
        default_hl = getattr(self.conf, "HOP_LIMIT", self.zone_radius * 2)
        pNew.hopLimit = getattr(pNew, "hopLimit", default_hl)

        self.verboseprint(
            "At time", round(self.env.now, 3),
            "node", self.nodeid,
            "flooding ZRP DATA packet", pNew.seq,
            "towards dest", destId,
            "(IERP not implemented yet)",
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
            # managed flooding TTL inside zone
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

        # at the receiver, distance is at least 1 hop from source
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
    # IERP / BRP and other ZRP parts – to be implemented later
    # =====================================================================

    def initiate_route_discovery(self, destId):
        # Placeholder for future IERP (interzone) logic
        pass

    def handle_ierp_rreq(self, packet):
        pass

    def handle_ierp_rrep(self, packet):
        pass

    def handle_zrp_control(self, packet):
        if isinstance(packet, MeshPacket_ZRP):
            if packet.packet_type == "IARP":
                self.handle_iarp(packet)
            elif packet.packet_type == "IERP":
                # Hook for future IERP control packets
                pass

    # =====================================================================
    # RECEIVE WITH MANAGED FLOODING FOR IARP + DATA
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
                # IARP packets: managed flooding with zone radius
                # ==================================================
                if isinstance(packet, MeshPacket_ZRP) and packet.packet_type == "IARP":
                    current_hops = getattr(packet, "hop_count", 0)

                    # Stop if already at or beyond zone radius
                    if current_hops >= self.zone_radius:
                        self.verboseprint(
                            "At time", round(self.env.now, 3),
                            "node", self.nodeid,
                            "dropped IARP packet", packet.seq,
                            "from", packet.origTxNodeId,
                            "because hop_count", current_hops,
                            ">= ZRP_ZONE_RADIUS", self.zone_radius,
                        )
                        continue

                    # Update local IARP table
                    self.handle_iarp(packet)

                    # Managed flood inside zone
                    remaining = getattr(packet, "hopLimit", self.zone_radius)
                    if (
                        packet.destId == NODENUM_BROADCAST
                        and not self.isClientMute
                        and remaining > 0
                        and current_hops + 1 < self.zone_radius
                    ):
                        pNew = MeshPacket_ZRP(
                            self.conf,
                            self.nodes,
                            origTxNodeId=packet.origTxNodeId, # keep original source
                            destId=NODENUM_BROADCAST,
                            txNodeId=self.nodeid,
                            packetLen=packet.packetLen,
                            seq=packet.seq,
                            genTime=packet.genTime,
                            wantAck=False,
                            isAck=False,
                            requestId=None,
                            txTime=self.env.now,
                            verboseprint=self.verboseprint,
                            packet_type="IARP",
                            iarp_seq_num=packet.iarp_seq_num,
                            hop_count=current_hops + 1,
                        )
                        pNew.hopLimit = remaining - 1

                        self.packets.append(pNew)
                        self.env.process(self.transmit(pNew))
                        self.verboseprint(
                            "At time", round(self.env.now, 3),
                            "node", self.nodeid,
                            "rebroadcasted IARP packet", pNew.seq,
                            "origin", packet.origTxNodeId,
                            "hop_count", pNew.hop_count,
                            "hopLimit", pNew.hopLimit,
                        )

                    # IARP is control-only, no data / ACK handling
                    continue

                # ==================================================
                # Data / ACK handling (non-IARP)
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
                        # ACK as ZRP-DATA packet (packet_type=None)
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
                        ackReceived = True
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
                        realAckReceived = True
                        sentPacket.ackReceived = True
