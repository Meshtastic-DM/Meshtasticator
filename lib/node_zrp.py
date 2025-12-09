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
        """
        Handle an incoming IARP packet.

        Minimal behavior for now:
          - Treat the sender as a 1-hop neighbor inside the zone.
          - Update or insert an IARPEntry for that neighbor using seq_num.
          - You can extend this later to carry multiple entries in the IARP payload.
        """
        # Ignore non-IARP packets
        if not isinstance(packet, MeshPacket_ZRP) or packet.packet_type != "IARP":
            return

        src = packet.origTxNodeId
        distance = packet.hop_count  # from this node to the origin of the IARP (one hop)
        seq_num = packet.iarp_seq_num
        now = self.env.now

        existing = self.iarp_table.get(src, None)

        # Update rule: newer sequence number OR same seq but shorter distance
        if (
            existing is None
            or seq_num > existing.seq_num
            or (seq_num == existing.seq_num and distance < existing.distance)
        ):
            self.iarp_table[src] = IARPEntry(
                destId=src,
                nextHop=packet.txNodeId,  # neighbor from which we heard this
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
