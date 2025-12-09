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

    For now:
      - Implements ONLY IARP (intrazone proactive routing).
      - Starts a periodic IARP update process.
      - Provides a handler to process incoming IARP packets.

    IERP / BRP and full interzone routing are left as empty stubs,
    to be implemented step by step.
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

        # IARP periodic update interval (ms) – you can tune this
        # Default: 5 minutes if not specified in config
        self.iarp_period_msec = getattr(self.conf, "IARP_PERIOD_MSEC", 5 * 60 * 1000)

        # Start periodic IARP process
        self.env.process(self._iarp_periodic_process())

    # =====================================================================
    # IARP: Intrazone proactive routing
    # =====================================================================

    def _iarp_periodic_process(self):
        """
        Periodically broadcast IARP updates within the zone.

        For now, we keep it very simple:
          - Every iarp_period_msec, this node sends an IARP packet
            advertising itself with distance 0.
          - You can later extend this to piggy-back full/partial tables.
        """
        while True:
            nextGen = self.get_next_time(self.iarp_period_msec)
            if nextGen < 0:
                break
            yield self.env.timeout(nextGen)

            # Increment local IARP sequence number
            self.iarp_seq_num += 1

            # Make a logical message record (optional; keeps your logs consistent)
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
                hop_count=0,
            )

            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "broadcasting IARP update with seq", self.iarp_seq_num,
            )

            self.packets.append(p)
            self.env.process(self.transmit(p))

    def handle_iarp(self, packet: MeshPacket_ZRP):
        if not isinstance(packet, MeshPacket_ZRP) or packet.packet_type != "IARP":
            return

        src = packet.origTxNodeId
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
        """
        Helper to inspect IARP table (for debug / logging).
        Returns a serializable dict similar to AODV's get_route_table.
        """
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
        """
        Placeholder for IERP RREQ logic.
        Implement later.
        """
        pass

    def handle_ierp_rreq(self, packet):
        """
        Placeholder for handling IERP RREQ packets.
        Implement later.
        """
        pass

    def handle_ierp_rrep(self, packet):
        """
        Placeholder for handling IERP RREP packets.
        Implement later.
        """
        pass

    def handle_zrp_control(self, packet):
        """
        Generic ZRP control dispatcher.
        You can call this from your receive() override later.

        For now:
          - If IARP packet → handle_iarp()
          - IERP packets → left empty
        """
        if isinstance(packet, MeshPacket_ZRP):
            if packet.packet_type == "IARP":
                self.handle_iarp(packet)
            elif packet.packet_type == "IERP":
                # Later: dispatch to handle_ierp_rreq / handle_ierp_rrep
                pass
    

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
                        "At time",
                        round(self.env.now, 3),
                        "node",
                        self.nodeid,
                        "started receiving packet",
                        packet.seq,
                        "from",
                        packet.txNodeId,
                    )
                    packet.onAirToN[self.nodeid] = False
                    self.isReceiving.append(True)
                else:
                    # was transmitting → cannot receive
                    self.verboseprint(
                        "At time",
                        round(self.env.now, 3),
                        "node",
                        self.nodeid,
                        "was transmitting, so could not receive packet",
                        packet.seq,
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
                        "At time",
                        round(self.env.now, 3),
                        "node",
                        self.nodeid,
                        "could not decode packet.",
                    )
                    continue

                # Drop if hop_count >= 7
                if hasattr(packet, "hop_count") and packet.hop_count >= 7:
                    self.verboseprint(
                        "At time",
                        round(self.env.now, 3),
                        "node",
                        self.nodeid,
                        "dropped packet",
                        packet.seq,
                        "due to hop_count >= 7",
                    )
                    continue

                packet.receivedAtN[self.nodeid] = True
                self.verboseprint(
                    "At time",
                    round(self.env.now, 3),
                    "node",
                    self.nodeid,
                    "received packet",
                    packet.seq,
                    "with delay",
                    round(self.env.now - packet.genTime, 2),
                )
                self.delays.append(self.env.now - packet.genTime)

                # ----------------- ZRP control handling (IARP only for now) -----------------
                if isinstance(packet, MeshPacket_ZRP) and packet.packet_type == "IARP":
                    # Only IARP implemented now
                    self.handle_iarp(packet)
                    # No ACK, no further data processing for IARP
                    continue

                # ----------------- Data / ACK handling -----------------
                if packet.destId == self.nodeid or packet.destId == NODENUM_BROADCAST:
                    # Generate ACK if required
                    if not packet.isAck and packet.wantAck:
                        self.messageSeq["val"] += 1
                        messageSeq = self.messageSeq["val"]
                        self.messages.append(
                            MeshMessage(
                                self.nodeid, packet.origTxNodeId, self.env.now, messageSeq
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
                            "At time",
                            round(self.env.now, 3),
                            "node",
                            self.nodeid,
                            "sent ACK for packet",
                            packet.seq,
                            "to",
                            packet.origTxNodeId,
                        )

                    self.verboseprint(
                        "At time",
                        round(self.env.now, 3),
                        "node",
                        self.nodeid,
                        "received packet",
                        packet.seq,
                        "from",
                        packet.origTxNodeId,
                    )

                    if not packet.isAck:
                        orginTxNodeId = packet.origTxNodeId
                        orginTxNode = None
                        for n in self.nodes:
                            if n.nodeid == orginTxNodeId:
                                orginTxNode = n
                                break

                        # ----------------- Sensor → Control center stats -----------------
                        if orginTxNode and orginTxNode.simRole == "Sensor":
                            self.verboseprint(
                                "At time",
                                round(self.env.now, 3),
                                "node",
                                self.nodeid,
                                "is a Control node receiving a packet from Sensor node",
                                orginTxNodeId,
                            )
                            if packet.seq not in self.SensorPacketsReceived:
                                self.SensorPacketsReceived[packet.seq] = 0
                                if packet.origTxNodeId not in self.SensorPacketsReceivedOrigId:
                                    self.SensorPacketsReceivedOrigId[
                                        packet.origTxNodeId
                                    ] = {}
                                self.SensorPacketsReceivedOrigId[packet.origTxNodeId][
                                    packet.seq
                                ] = 0
                                if packet.origTxNodeId not in self.SensorPacketsDelays:
                                    self.SensorPacketsDelays[packet.origTxNodeId] = []
                                self.SensorPacketsDelays[packet.origTxNodeId].append(
                                    self.env.now - packet.genTime
                                )
                            self.SensorPacketsReceived[packet.seq] += 1
                            self.SensorPacketsReceivedOrigId[packet.origTxNodeId][
                                packet.seq
                            ] += 1

                        # ----------------- Control_Center broadcast handling -----------------
                        elif orginTxNode and orginTxNode.simRole == "Control_Center":
                            if packet.seq not in self.BroadcastPacketsReceived:
                                self.BroadcastPacketsReceived[packet.seq] = 0
                                # First time seeing this broadcast → rebroadcast (subject to hop_count)
                                if not self.isClientMute:
                                    self.verboseprint(
                                        "At time",
                                        round(self.env.now, 3),
                                        "node",
                                        self.nodeid,
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
                                        "At time",
                                        round(self.env.now, 3),
                                        "node",
                                        self.nodeid,
                                        "rebroadcasted broadcast packet",
                                        pNew.seq,
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

                        # ----------------- DM unicast stats -----------------
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
                    # Not for me and not pure broadcast delivery case → optional forwarding
                    if not self.isClientMute:
                        self.verboseprint(
                            "At time",
                            round(self.env.now, 3),
                            "node",
                            self.nodeid,
                            "rebroadcasts received packet",
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
                            "At time",
                            round(self.env.now, 3),
                            "node",
                            self.nodeid,
                            "rebroadcasted packet",
                            pNew.seq,
                            "to",
                            pNew.destId,
                        )
                    else:
                        self.verboseprint(
                            "At time",
                            round(self.env.now, 3),
                            "node",
                            self.nodeid,
                            "dropped packet",
                            packet.seq,
                            "because client is muted",
                        )

                # ----------------- ACK bookkeeping for queue -----------------
                for sentPacket in self.packets:
                    # implicit ACK (same seq still in queue)
                    if sentPacket.txNodeId == self.nodeid and sentPacket.seq == packet.seq:
                        self.verboseprint(
                            "At time",
                            round(self.env.now, 3),
                            "node",
                            self.nodeid,
                            "received implicit ACK for message in queue.",
                        )
                        ackReceived = True
                        sentPacket.ackReceived = True
                    # real ACK
                    if (
                        sentPacket.origTxNodeId == self.nodeid
                        and packet.isAck
                        and sentPacket.seq == packet.requestId
                    ):
                        self.verboseprint(
                            "At time",
                            round(self.env.now, 3),
                            "node",
                            self.nodeid,
                            "received real ACK.",
                        )
                        realAckReceived = True
                        sentPacket.ackReceived = True

