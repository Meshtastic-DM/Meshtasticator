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

        # For IARP duplicate suppression
        self.seen_iarp = set()  # (origTxNodeId, iarp_seq_num)

        # ------------- IERP state ----------------
        self.ierp_seq_num = 0
        self.ierp_table: dict[int, IARPEntry] = {}  # optional coarse inter-zone info
        self.processed_ierp_rreq = set()  # (origTxNodeId, ierp_id)
        self.processed_ierp_rrep = set()  # (origTxNodeId, ierp_id)
        self.hopLimit = getattr(self.conf, "hopLimit", 3)

        # Start periodic IARP process
        self.env.process(self._iarp_periodic_process())

    # =====================================================================
    # Helpers for ZRP / IERP
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
            default_hl = getattr(self, "hopLimit", self.zone_radius)
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
            default_hl = getattr(self, "hopLimit", self.zone_radius)
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
            "→ triggering IERP route discovery.",
        )

        # Queue data for later delivery when RREP arrives (optional but useful)
        self.pending_ierp.setdefault(destId, []).append(base_packet)

        # Start IERP RREQ bordercast
        self.initiate_route_discovery(destId)

        # For now we do NOT flood the data blindly; it will be sent once
        # a route is discovered (when you implement RREP wiring).
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

        # ----- duplicate suppression -----
        key = (src, packet.iarp_seq_num)
        if key in self.seen_iarp:
            # already processed this IARP from this origin + seq
            return
        self.seen_iarp.add(key)
        # ---------------------------------

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

    def get_ierp_table(self):
        info = {}
        for destId, entry in self.ierp_table.items():
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
        IERP RREQ bordercast.

        - destId: final query destination (application node).
        - Each IERP-RREQ copy:
            * destId      = peripheral node (distance == zone_radius).
            * ierp_destId = real query destination.
        - covered_nodes:
            * At origin: all nodes in my intrazone set (including myself).
        """
        peripherals = self.get_peripheral_neighbors()
        if not peripherals:
            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "has no peripheral neighbors for IERP; discovery aborted.",
            )
            return

        self.ierp_seq_num += 1
        ierp_id = self.ierp_seq_num

        # Build initial covered_nodes = my intrazone (including me)
        covered_nodes = list(self.get_intrazone_nodes_set())

        for entry in peripherals:
            rreq = MeshPacket_ZRP(
                self.conf,
                self.nodes,
                origTxNodeId=self.nodeid,    # IERP origin
                destId=entry.destId,         # peripheral node in my zone
                txNodeId=self.nodeid,
                packetLen=0,                 # overridden by MeshPacket_ZRP for IERP
                seq=ierp_id,                 # can reuse for control ID
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
                ierp_destId=destId,          # real query destination
                hop_count=0,
                covered_nodes=covered_nodes,
            )

            # RREQ hopLimit – use config if available, else a small radius (e.g. 7)
            base_hl = getattr(self, "hopLimit", 3)
            rreq.hopLimit = getattr(self.conf, "ZRP_IERP_MAX_TTL", base_hl)


            # For PHY, we can treat this as "unicast to next_hop" (peripheral next hop)
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
                    "due to hopLimit < 0",
                )
                return

        # Optional global bound to prevent crazy expansion
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

        # -------------------------------
        # Check termination cases
        # -------------------------------
        # If I am the query destination, you'd generate RREP here
        if self.nodeid == query_dest:
            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "is query dest", query_dest,
                "→ would send IERP RREP (not yet implemented).",
            )
            # TODO: send IERP RREP back towards packet.origTxNodeId
            return

        # If IARP has route to query_dest, I can act as proxy responder
        iarp_entry = self.iarp_table.get(query_dest)
        if iarp_entry and iarp_entry.distance <= self.zone_radius:
            self.verboseprint(
                "At time", round(self.env.now, 3),
                "node", self.nodeid,
                "has IARP route to query dest", query_dest,
                "→ would send IERP RREP (not yet implemented).",
            )
            # TODO: send IERP RREP based on IARP route
            return

        # Store coarse inter-zone info about the origin once RREQ has gone
        # beyond my intrazone radius (i.e., inter-zone information).
        if packet.hop_count > self.zone_radius:
            origin = packet.origTxNodeId
            existing = self.ierp_table.get(origin)

            # Use ierp_id as a "sequence number" for freshness
            if existing is None or packet.ierp_id > existing.seq_num:
                self.ierp_table[origin] = IARPEntry(
                    destId=origin,
                    # To reach the origin zone, send towards the node we just
                    # received this RREQ from (reverse direction).
                    nextHop=packet.txNodeId,
                    distance=packet.hop_count,
                    seq_num=packet.ierp_id,
                    last_updated=self.env.now,
                )
                self.verboseprint(
                    "At time", round(self.env.now, 3),
                    "node", self.nodeid,
                    "updated IERP entry for origin", origin,
                    "via nextHop", packet.txNodeId,
                    "distance", packet.hop_count,
                    "ierp_id", packet.ierp_id,
                )


        # -------------------------------
        # Bordercast further to peripherals not in covered_nodes
        # -------------------------------
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
                destId=entry.destId,     # new peripheral node
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
                ierp_destId=query_dest,
                hop_count=packet.hop_count,
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


    def handle_ierp_rrep(self, packet: MeshPacket_ZRP):
        if packet.packet_type != "IERP" or packet.ierp_type != "RREP":
            return

        key = (packet.origTxNodeId, packet.ierp_id)
        if key in self.processed_ierp_rrep:
            self.verboseprint("ZRP: duplicate IERP RREP dropped", key)
            return
        self.processed_ierp_rrep.add(key)

        self.verboseprint(
            "At time", round(self.env.now, 3),
            "node", self.nodeid,
            "received IERP RREP", packet.ierp_id,
            "from", packet.txNodeId,
            "for dest", packet.ierp_destId,
            "→ TODO: install route + forward / flush pending_ierp.",
        )
        # TODO:
        # - install self.ierp_table[packet.ierp_destId]
        # - if self.nodeid == original source → send pending_ierp[destId]
        # - else forward RREP towards origTxNodeId



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
                # IERP control packets – RREQ / RREP
                # ==================================================
                if isinstance(packet, MeshPacket_ZRP) and packet.packet_type == "IERP":

                    # 1) If I'm the intended border / query node → process control
                    if packet.destId == self.nodeid:
                        if packet.ierp_type == "RREQ":
                            self.handle_ierp_rreq(packet)
                        elif packet.ierp_type == "RREP":
                            self.handle_ierp_rrep(packet)
                        # Done as control, don't go to DATA or generic forwarding
                        continue

                    # 2) Transit hop: obey next_hop if present
                    if (
                        hasattr(packet, "next_hop")
                        and packet.next_hop is not None
                        and packet.next_hop != self.nodeid
                    ):
                        # I just overheard it, not my turn to forward
                        self.verboseprint(
                            "At time", round(self.env.now, 3),
                            "node", self.nodeid,
                            "ignores IERP packet", packet.seq,
                            "because next_hop is", packet.next_hop,
                        )
                        continue

                    # 3) Compute hop_count as seen *after* this node
                    prev_hops = getattr(packet, "hop_count", 0)
                    curr_hops = prev_hops + 1   # distance from origin including this node

                    # 3b) If beyond my zone, install coarse IERP entry for the origin
                    if curr_hops > self.zone_radius:
                        origin = packet.origTxNodeId
                        existing = self.ierp_table.get(origin)
                        if existing is None or packet.ierp_id > existing.seq_num:
                            self.ierp_table[origin] = IARPEntry(
                                destId=origin,
                                nextHop=packet.txNodeId,      # send back toward where RREQ came from
                                distance=curr_hops,
                                seq_num=packet.ierp_id,
                                last_updated=self.env.now,
                            )
                            self.verboseprint(
                                "At time", round(self.env.now, 3),
                                "node", self.nodeid,
                                "updated IERP entry for origin", origin,
                                "via nextHop", packet.txNodeId,
                                "distance", curr_hops,
                                "ierp_id", packet.ierp_id,
                            )

                    # 4) TTL / hop-limit handling
                    hl = getattr(packet, "hopLimit", None)
                    if hl is not None:
                        if hl <= 0:
                            self.verboseprint(
                                "At time", round(self.env.now, 3),
                                "node", self.nodeid,
                                "drops IERP packet", packet.seq,
                                "due to hopLimit <= 0",
                            )
                            continue
                        hl -= 1

                    # 5) Route towards this packet.destId using my IARP
                    route = self.iarp_table.get(packet.destId)
                    if route is None or route.distance > self.zone_radius:
                        self.verboseprint(
                            "At time", round(self.env.now, 3),
                            "node", self.nodeid,
                            "has no IARP route for IERP dest", packet.destId,
                            "→ dropping.",
                        )
                        continue

                    # 6) Forward as IERP again (NOT DATA)
                    fwd = MeshPacket_ZRP(
                        self.conf,
                        self.nodes,
                        origTxNodeId=packet.origTxNodeId,
                        destId=packet.destId,
                        txNodeId=self.nodeid,
                        packetLen=packet.packetLen,
                        seq=packet.seq,
                        genTime=packet.genTime,
                        wantAck=False,
                        isAck=False,
                        requestId=None,
                        txTime=self.env.now,
                        verboseprint=self.verboseprint,
                        packet_type="IERP",            # keep as IERP
                        iarp_seq_num=None,
                        ierp_type=packet.ierp_type,    # RREQ or RREP
                        ierp_id=packet.ierp_id,
                        hop_count=curr_hops,
                    )

                    # preserve extended fields
                    fwd.ierp_destId   = getattr(packet, "ierp_destId", None)
                    fwd.covered_nodes = getattr(packet, "covered_nodes", None)
                    fwd.hopLimit      = hl
                    fwd.next_hop      = route.nextHop

                    self.verboseprint(
                        "At time", round(self.env.now, 3),
                        "node", self.nodeid,
                        "forwards IERP", fwd.ierp_type,
                        "id", fwd.ierp_id,
                        "towards dest", fwd.destId,
                        "via nextHop", fwd.next_hop,
                        "hop_count", fwd.hop_count,
                        "hopLimit", fwd.hopLimit,
                    )
                    self.packets.append(fwd)
                    self.env.process(self.transmit(fwd))

                    # IMPORTANT: do not let IERP fall into DATA / forwarding sections
                    continue




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
                    # ------------ deliver to local app + generate ACK ------------
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

                    # ------------ application-facing stats (unchanged) ------------
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

                        # Control_Center broadcast handling (only for *broadcast* app packets)
                        elif orginTxNode and orginTxNode.simRole == "Control_Center":
                            if packet.seq not in self.BroadcastPacketsReceived:
                                self.BroadcastPacketsReceived[packet.seq] = 0
                                # For app-level broadcast, we still flood
                                if not self.isClientMute and packet.destId == NODENUM_BROADCAST:
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

                # ==================================================
                # FORWARDING (not for me, not broadcast-to-app)
                # ==================================================
                else:
                    # ------------- UNICAST: use IARP hop-by-hop -------------
                    if packet.destId != NODENUM_BROADCAST:
                        # If this packet has a next_hop and it's not me → ignore
                        if (
                            hasattr(packet, "next_hop")
                            and packet.next_hop is not None
                            and packet.next_hop != self.nodeid
                        ):
                            self.verboseprint(
                                "At time", round(self.env.now, 3),
                                "node", self.nodeid,
                                "ignores packet", packet.seq,
                                "because next_hop is", packet.next_hop,
                            )
                            continue

                        if self.isClientMute:
                            self.verboseprint(
                                "At time", round(self.env.now, 3),
                                "node", self.nodeid,
                                "dropped unicast packet", packet.seq,
                                "because client is muted",
                            )
                            continue

                        # Decrement hopLimit if present
                        hl = getattr(packet, "hopLimit", None)
                        if hl is not None:
                            if hl <= 0:
                                self.verboseprint(
                                    "At time", round(self.env.now, 3),
                                    "node", self.nodeid,
                                    "drops unicast packet", packet.seq,
                                    "due to hopLimit <= 0",
                                )
                                continue
                            hl -= 1

                        # Look up next hop from *my* IARP table
                        route = self.iarp_table.get(packet.destId)
                        if route is None or route.distance > self.zone_radius:
                            # No intrazone route → optional fallback: drop or flood
                            self.verboseprint(
                                "At time", round(self.env.now, 3),
                                "node", self.nodeid,
                                "has no IARP route for dest", packet.destId,
                                "→ dropping (or you can flood here if you want).",
                            )
                            continue

                        # Build next hop packet
                        fwd = MeshPacket_ZRP(
                            self.conf,
                            self.nodes,
                            packet.origTxNodeId,      # still original source
                            packet.destId,            # final destination
                            self.nodeid,              # current transmitter
                            packet.packetLen,
                            packet.seq,
                            packet.genTime,
                            packet.wantAck,
                            packet.isAck,
                            packet.requestId,
                            self.env.now,
                            self.verboseprint,
                            None,                     # DATA
                            None,
                            None,
                            None,
                            getattr(packet, "hop_count", 0) + 1,
                        )
                        fwd.next_hop = route.nextHop
                        if hl is not None:
                            fwd.hopLimit = hl

                        self.packets.append(fwd)
                        self.env.process(self.transmit(fwd))
                        self.verboseprint(
                            "At time", round(self.env.now, 3),
                            "node", self.nodeid,
                            "forwarded unicast packet", fwd.seq,
                            "towards dest", fwd.destId,
                            "via nextHop", fwd.next_hop,
                            "hop_count", fwd.hop_count,
                            "hopLimit", getattr(fwd, 'hopLimit', None),
                        )

                    # ------------- BROADCAST (non-app) – still flood, but with hopLimit -------------
                    else:
                        if not self.isClientMute:
                            # hopLimit check for generic broadcast
                            hl = getattr(packet, "hopLimit", None)
                            if hl is not None:
                                if hl <= 0:
                                    self.verboseprint(
                                        "At time", round(self.env.now, 3),
                                        "node", self.nodeid,
                                        "drops broadcast packet", packet.seq,
                                        "due to hopLimit <= 0",
                                    )
                                    continue
                                hl -= 1

                            self.verboseprint(
                                "At time", round(self.env.now, 3),
                                "node", self.nodeid,
                                "rebroadcasts received broadcast packet", packet.seq,
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
                                None,   # DATA broadcast
                                None,
                                None,
                                None,
                                getattr(packet, "hop_count", 0) + 1,
                            )
                            if hl is not None:
                                pNew.hopLimit = hl

                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                            self.verboseprint(
                                "At time", round(self.env.now, 3),
                                "node", self.nodeid,
                                "rebroadcasted broadcast packet", pNew.seq,
                                "to", pNew.destId,
                            )
                        else:
                            self.verboseprint(
                                "At time", round(self.env.now, 3),
                                "node", self.nodeid,
                                "dropped broadcast packet", packet.seq,
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
