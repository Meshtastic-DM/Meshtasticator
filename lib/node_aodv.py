from    lib.node import MeshNode
from    lib.packet_aodv import MeshPacket_AODV
from    lib.packet import NODENUM_BROADCAST, MeshMessage
import  simpy
import  random
from lib.common import calc_dist, find_random_position
from lib.mac import set_transmit_delay, get_retransmission_msec
from lib.packet_aodv import MeshPacket_AODV
from lib.phy import check_collision, is_channel_active, airtime

class MeshNode_AODV(MeshNode):
    """
    This class extends MeshNode to implement AODV routing protocol functionality.
    It overrides necessary methods and adds new attributes and methods specific to AODV.
    """
    def __init__(self, conf, nodes, env, bc_pipe, nodeid, period, messages, packetsAtN, packets, delays, nodeConfig, messageSeq, verboseprint):
        super().__init__( conf, nodes, env, bc_pipe, nodeid, period, messages, packetsAtN, packets, delays, nodeConfig, messageSeq, verboseprint)
        self.routing_table = {}  # key: destination nodeId, value: next hop nodeId
        self.rreq_id_counter = 0  # Counter for generating unique RREQ IDs
        self.pending_rreq = {}  # key: (destId), value: list of packets waiting for route
        self.seq_num = 0  # Sequence number for this node
        #self.messages = []  # List to store generated messages
        self.routing_table: dict[int, RouteEntry] = {}
        self.transmitter = simpy.Resource(env, 1)
        self.processed_rreq = set()  # Set to track processed RREQs to avoid loops
        self.processed_rrep = set()  # Set to track processed RREPs to avoid loops
        self.forwarded_rrep = set()  # Set to track forwarded RREPs to avoid loops
        
    
    def aodv_reliable_retransmit(self, p):
        """
        Retransmit logic for AODV unicast packets that wantAck.
        Stops when ACK is observed in self.packets, or retries exhausted.
        IMPORTANT: p must be the "original" packet (same seq) that we track.
        """
        if not getattr(p, "wantAck", False):
            return

        # Don't retransmit broadcasts
        if p.destId == NODENUM_BROADCAST:
            return

        # Max retry count (how many *additional* attempts after first send)
        max_retx = getattr(self.conf, "maxRetransmission", 3)

        # We count remaining retx attempts in the packet object (optional)
        remaining = getattr(p, "retransmissions", max_retx)
        p.retransmissions = remaining

        while p.wantAck:
            # Wait before checking/retrying
            retransmissionMsec = get_retransmission_msec(self, p)
            yield self.env.timeout(retransmissionMsec)

            # Check if ack already received for this packet seq
            ack_received = False

            # Look through sent packets and see if any entry for this seq got acked
            for packetSent in self.packets:
                if packetSent.origTxNodeId == self.nodeid and packetSent.seq == p.seq:
                    if getattr(packetSent, "ackReceived", False):
                        ack_received = True
                        break

            if ack_received:
                self.verboseprint(
                    "[AODV RETX EXIT ACK]",
                    "time", round(self.env.now, 3),
                    "| node", self.nodeid,
                    "| seq", p.seq,
                )
                break

            # No ACK → retransmit if we still have retries
            if p.retransmissions > 0:
                # Must re-evaluate next hop (route might have changed)
                nh = None
                if p.destId in self.routing_table and self.routing_table[p.destId].valid:
                    nh = self.routing_table[p.destId].nextHop

                # If no route now, stop (or you can trigger a new RREQ)
                if nh is None:
                    self.verboseprint(
                        "[AODV RETX EXIT NO ROUTE]",
                        "time", round(self.env.now, 3),
                        "| node", self.nodeid,
                        "| seq", p.seq,
                        "| dest", p.destId,
                    )
                    break

                pNew = MeshPacket_AODV(
                    self.conf, self.nodes,
                    p.origTxNodeId, p.destId,
                    self.nodeid,              # txNodeId = me
                    p.packetLen,
                    p.seq,
                    p.genTime,
                    p.wantAck,
                    False,                    # isAck
                    getattr(p, "rreq_id", None),
                    self.env.now,
                    self.verboseprint,
                    data=getattr(p, "data", None),
                    rreq_id=getattr(p, "rreq_id", None)
                )

                # Copy fields used by your forwarding logic
                pNew.hopLimit = getattr(p, "hopLimit", 10)
                pNew.next_hop = nh
                pNew.hop_count = getattr(p, "hop_count", 0)
                pNew.ttl = getattr(p, "ttl", 64)
                pNew.is_rreq = getattr(p, "is_rreq", False)
                pNew.is_rrep = getattr(p, "is_rrep", False)
                pNew.is_rerr = getattr(p, "is_rerr", False)
                pNew.is_sdn_update = getattr(p, "is_sdn_update", False)

                # decrement remaining attempts
                p.retransmissions -= 1
                pNew.retransmissions = p.retransmissions

                self.verboseprint(
                    "[AODV RETX SEND]",
                    "time", round(self.env.now, 3),
                    "| node", self.nodeid,
                    "| seq", p.seq,
                    "| remaining", pNew.retransmissions,
                    "| next_hop", nh,
                )

                self.packets.append(pNew)
                self.env.process(self.transmit(pNew))

            else:
                self.verboseprint(
                    "[AODV RETX EXIT FAIL]",
                    "time", round(self.env.now, 3),
                    "| node", self.nodeid,
                    "| seq", p.seq,
                )
                break


    def send_packet(self, destId,data = None, wantAck=True,is_sdn_update=False):
        plen = 20
        self.seq_num += 1
        self.messageSeq["val"] += 1
        messageSeq = self.messageSeq["val"]
        self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))
        p = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, destId, self.nodeid, plen, messageSeq, self.env.now, wantAck, False, None, self.env.now, self.verboseprint,data=data)
        p.is_sdn_update = is_sdn_update
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'preparing to send packet', p.seq, 'to', destId,'is_sdn_update:',is_sdn_update)
        if destId != NODENUM_BROADCAST:
            if destId in self.routing_table and self.routing_table[destId].valid and self.routing_table[destId].lifeTime > self.env.now:
                pNew = MeshPacket_AODV(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, p.isAck, None, self.env.now, self.verboseprint,data=p.data)
                pNew.is_sdn_update = p.is_sdn_update
                pNew.hopLimit = p.hopLimit - 1
                pNew.next_hop = self.routing_table[destId].nextHop if destId in self.routing_table else None
                if pNew.next_hop is None:
                    self.verboseprint(
                        "[DATA SEND NEXT_HOP_NONE]",
                        "time", round(self.env.now, 3),
                        "| node", self.nodeid,
                        "| seq", pNew.seq,
                        "| dest", destId,
                        "| route_exists", (destId in self.routing_table),
                        "| valid", (self.routing_table[destId].valid if destId in self.routing_table else None),
                        "| lifeTime", (self.routing_table[destId].lifeTime if destId in self.routing_table else None),
                        "| now", self.env.now,
                        "| entry_nextHop", (self.routing_table[destId].nextHop if destId in self.routing_table else None),
                    )

                self.packets.append(pNew)
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is sending packet', pNew.seq, 'to', pNew.destId, 'via next hop', self.routing_table[destId].nextHop)
                self.env.process(self.transmit(pNew))
            else:
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'no valid route to', destId, 'initiating route discovery')
                # Initiate route discovery
                p.queued_no_route = True
                self.initiate_route_discovery(destId)
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is initiating route discovery for', destId)
                # Store the packet to be sent once the route is discovered
                if (destId) not in self.pending_rreq:
                    self.pending_rreq[(destId)] = []
                self.pending_rreq[(destId)].append(p)
        else:
            # Broadcast packet
            pNew = MeshPacket_AODV(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, p.isAck, None, self.env.now, self.verboseprint)
            pNew.hopLimit = p.hopLimit - 1
            pNew.is_sdn_update = p.is_sdn_update
            pNew.next_hop = None
            pNew.data = p.data
            pNew.is_rerr = p.is_rerr
            pNew.is_rrep = p.is_rrep
            pNew.is_rreq = p.is_rreq
            pNew.hop_count = p.hop_count
            self.packets.append(pNew)
            if self.simRole == "sdn_node":
                self.verboseprint('At time', round(self.env.now, 3), 'SDN controller node', self.nodeid, 'broadcasted SDN node announcement packet', pNew.seq)
            else:
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'broadcasting packet', pNew.seq)
            self.env.process(self.transmit(pNew))
        #self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))
        return p

    def initiate_route_discovery(self, destId):
        self.rreq_id_counter += 1
        rreq_id = self.rreq_id_counter
        self.messageSeq["val"] += 1
        messageSeq = self.messageSeq["val"]
        self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))
        rreq_packet = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, destId, self.nodeid, 10, messageSeq, self.env.now, False, False, rreq_id, self.env.now, self.verboseprint,rreq_id=rreq_id)
        rreq_packet.is_rreq = True
        rreq_packet.is_rrep = False
        rreq_packet.is_rerr = False
        rreq_packet.hop_count = 0
        rreq_packet.ttl = 64  # Initial TTL value for RREQ
        rreq_packet.hopLimit =7
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'created RREQ for', destId, 'with RREQ_ID', rreq_id)
        self.packets.append(rreq_packet)
        self.env.process(self.transmit(rreq_packet))
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'transmitted RREQ for', destId, 'with RREQ_ID', rreq_id)

    def handle_rreq(self, packet):
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received RREQ from', packet.origTxNodeId, 'for', packet.destId, 'RREQ_ID', packet.rreq_id)
        if packet.hopLimit <= 0:
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'dropped RREQ due to hop limit reached')
            return  # Drop the RREQ if hop limit is reached
        if not packet.is_rreq:
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'packet is not RREQ, ignoring')
            return  # Not a valid RREQ packet
        if packet.origTxNodeId == self.nodeid:
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is the source of RREQ, ignoring')
            return  # RREQ originated from this node, ignore
        # Check if this RREQ has been processed before
        if (packet.origTxNodeId, packet.rreq_id) in self.processed_rreq:
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'already processed RREQ from', packet.origTxNodeId, 'RREQ_ID', packet.rreq_id)
            return  # Already processed

        # Mark this RREQ as processed
        self.processed_rreq.add((packet.origTxNodeId, packet.rreq_id))
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'marked RREQ as processed from', packet.origTxNodeId, 'RREQ_ID', packet.rreq_id)

        # Update routing table with reverse route to the source
        if packet.origTxNodeId not in self.routing_table or not self.routing_table[packet.origTxNodeId].valid or \
           packet.hop_count + 1 < self.routing_table[packet.origTxNodeId].hopCount:
            # self.routing_table[packet.origTxNodeId] = RouteEntry(
            #     destId=packet.origTxNodeId,
            #     nextHop=packet.txNodeId,
            #     hopCount=packet.hop_count + 1,
            #     destSeqNum=packet.seq,
            #     valid=True,
            #     precursorList=[],
            #     lifeTime=self.env.now + 300000  
            # )
            self.update_routing_table(packet.origTxNodeId, packet.txNodeId, packet.hop_count + 1, packet.seq)
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'updated reverse route to', packet.origTxNodeId, 'via', packet.txNodeId)
        # If this node is the destination, send RREP    
        if packet.destId == self.nodeid:
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is destination for RREQ, sending RREP to', packet.origTxNodeId)
            self.send_rrep(packet)
        elif packet.ttl > 1:
            # Forward the RREQ
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'forwarding RREQ from', packet.origTxNodeId, 'for', packet.destId, 'RREQ_ID', packet.rreq_id)
            fwd_packet = MeshPacket_AODV(self.conf, self.nodes, packet.origTxNodeId, packet.destId, self.nodeid, 10, packet.seq, self.env.now, False, False, None, self.env.now, self.verboseprint, rreq_id=packet.rreq_id)
            fwd_packet.is_rreq = True
            fwd_packet.is_rrep = False
            fwd_packet.is_rerr = False
            fwd_packet.hop_count = packet.hop_count + 1
            fwd_packet.ttl = packet.ttl - 1
            fwd_packet.hopLimit = packet.hopLimit - 1
            fwd_packet.txNodeId = self.nodeid
            self.packets.append(fwd_packet)
            self.env.process(self.transmit(fwd_packet)) # Rebroadcast the RREQ
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasted RREQ for', packet.destId, 'RREQ_ID', packet.rreq_id)

    def send_rrep(self, rreq_packet):
        key = (rreq_packet.rreq_id, rreq_packet.origTxNodeId, rreq_packet.destId)
        if key in self.processed_rrep:
            self.verboseprint('AODV: duplicate RREP dropped', key)
            return
        self.processed_rrep.add(key)
        self.messageSeq["val"] += 1
        messageSeq = self.messageSeq["val"]
        self.messages.append(MeshMessage(self.nodeid, rreq_packet.origTxNodeId, self.env.now, messageSeq))
        rrep_packet = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, rreq_packet.origTxNodeId, self.nodeid, 10, messageSeq, self.env.now, False, False, None, self.env.now, self.verboseprint, rreq_id=rreq_packet.rreq_id)
        rrep_packet.is_rreq = False
        rrep_packet.is_rrep = True
        rrep_packet.is_rerr = False
        rrep_packet.hop_count = 0
        rrep_packet.ttl = 64  # Initial TTL value for RREP
        rrep_packet.hopLimit =5
        rrep_packet.next_hop = self.routing_table.get(rreq_packet.origTxNodeId).nextHop if rreq_packet.origTxNodeId in self.routing_table else None
        if rrep_packet.next_hop is None:
            self.verboseprint(
                "[RREP SEND NEXT_HOP_NONE]",
                "time", round(self.env.now, 3),
                "| node", self.nodeid,
                "| rreq_origin", rreq_packet.origTxNodeId,
                "| rreq_id", rreq_packet.rreq_id,
                "| reverse_route_exists", (rreq_packet.origTxNodeId in self.routing_table),
                "| reverse_valid", (self.routing_table[rreq_packet.origTxNodeId].valid if rreq_packet.origTxNodeId in self.routing_table else None),
                "| reverse_nextHop", (self.routing_table[rreq_packet.origTxNodeId].nextHop if rreq_packet.origTxNodeId in self.routing_table else None),
                "| rreq_txNodeId", getattr(rreq_packet, "txNodeId", None),
                "| hint", "Reverse route missing/invalid => RREP can't go back",
            )

        # Update routing table with forward route to the destination
        # self.routing_table[rreq_packet.origTxNodeId] = RouteEntry(
        #     destId=rreq_packet.origTxNodeId,
        #     nextHop=rreq_packet.txNodeId,
        #     hopCount=1,
        #     destSeqNum=rreq_packet.seq,
        #     valid=True,
        #     precursorList=[],
        #     lifeTime=self.env.now + 30000  # Example lifetime
        # )
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'sending RREP to', rreq_packet.origTxNodeId, 'via next hop', rreq_packet.txNodeId)
        self.packets.append(rrep_packet)
        self.env.process(self.transmit(rrep_packet))
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'transmitted RREP to', rreq_packet.origTxNodeId)

    def handle_rrep(self, packet):
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received RREP from', packet.txNodeId, 'for', packet.origTxNodeId)
        if not packet.is_rrep:
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'packet is not RREP, ignoring')
            return  # Not a valid RREP packet
        if packet.origTxNodeId == self.nodeid:
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is the source of RREQ, RREP reached destination')
            return  # RREP has reached the source
        key = (packet.origTxNodeId, packet.txNodeId,packet.seq)
        if key in self.forwarded_rrep:
            self.verboseprint('AODV: duplicate RREP dropped', key)
            return
        self.forwarded_rrep.add(key)
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'marked RREP as processed from', packet.txNodeId, 'for', packet.origTxNodeId)
        # Update routing table with forward route to the destination
        if packet.origTxNodeId not in self.routing_table or not self.routing_table[packet.origTxNodeId].valid or \
           packet.hop_count + 1 < self.routing_table[packet.origTxNodeId].hopCount:
            # self.routing_table[packet.origTxNodeId] = RouteEntry(
            #     destId=packet.origTxNodeId,
            #     nextHop=packet.txNodeId,
            #     hopCount=packet.hop_count + 1,
            #     destSeqNum=packet.seq,
            #     valid=True,
            #     precursorList=[],
            #     lifeTime=self.env.now + 3000000  # Example lifetime
            # )
            self.update_routing_table(packet.origTxNodeId, packet.txNodeId, packet.hop_count + 1, packet.seq)
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'updated forward route to', packet.origTxNodeId, 'via', packet.txNodeId)
        # If this node is the source of the RREQ, send pending packets
        if packet.destId == self.nodeid:
            key = (packet.origTxNodeId)
            if key in self.pending_rreq:
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'sending pending packets for', packet.origTxNodeId)
                for p in self.pending_rreq[key]:
                    pNew = MeshPacket_AODV(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, p.isAck, None, self.env.now, self.verboseprint,data=p.data)
                    pNew.is_sdn_update = p.is_sdn_update
                    pNew.next_hop = self.routing_table.get(p.destId).nextHop if p.destId in self.routing_table else None
                    self.packets.append(pNew)
                    self.env.process(self.transmit(pNew))
                    self.env.process(self.aodv_reliable_retransmit(pNew))
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'transmitted pending packet', pNew.seq, 'to', pNew.destId)
                del self.pending_rreq[key]
        elif packet.hopLimit > 1 and packet.next_hop == self.nodeid:
            # Forward the RREP
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'forwarding RREP for', packet.origTxNodeId)
            fwd_packet = MeshPacket_AODV(self.conf, self.nodes, packet.origTxNodeId, packet.destId, self.nodeid, 10, packet.seq, self.env.now, False, False, None, self.env.now, self.verboseprint, rreq_id=packet.rreq_id)
            fwd_packet.is_rreq = False
            fwd_packet.is_rrep = True
            fwd_packet.is_rerr = False
            fwd_packet.hop_count = packet.hop_count + 1
            fwd_packet.ttl = packet.ttl - 1
            fwd_packet.hopLimit = packet.hopLimit - 1
            nextHop = self.routing_table.get(packet.destId).nextHop if packet.destId in self.routing_table else None
            if nextHop is None:
                self.verboseprint(
                    "[RREP FWD NEXT_HOP_NONE]",
                    "time", round(self.env.now, 3),
                    "| node", self.nodeid,
                    "| rrep_seq", packet.seq,
                    "| rreq_id", packet.rreq_id,
                    "| rrep_dest(target)", packet.destId,
                    "| route_exists", (packet.destId in self.routing_table),
                    "| route_valid", (self.routing_table[packet.destId].valid if packet.destId in self.routing_table else None),
                    "| route_nextHop", (self.routing_table[packet.destId].nextHop if packet.destId in self.routing_table else None),
                    "| packet_next_hop_in", getattr(packet, "next_hop", None),
                    "| packet_txNodeId", getattr(packet, "txNodeId", None),
                    "| hint", "No route towards RREQ source => RREP will die here",
                )
            fwd_packet.next_hop = nextHop

            fwd_packet.txNodeId = self.nodeid
            self.packets.append(fwd_packet)
            self.env.process(self.transmit(fwd_packet)) # Rebroadcast the RREP
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasted RREP for', packet.origTxNodeId,'next hop',fwd_packet.next_hop)

    def handle_rerr(self, packet):
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received RERR for', packet.destId)
        if not packet.is_rerr:
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'packet is not RERR, ignoring')
            return  # Not a valid RERR packet
        # Invalidate the route to the unreachable destination
        if packet.destId in self.routing_table:
            self.routing_table[packet.destId].valid = False
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'invalidated route to', packet.destId)
        # Forward the RERR to precursors if any
        for precursor in self.routing_table.get(packet.destId, RouteEntry(None, None, None, None, False, [], None)).precursorList:
            rerr_packet = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, precursor, self.nodeid, 10, self.seq_num, self.env.now, False, False, None, self.env.now, self.verboseprint)
            rerr_packet.is_rreq = False
            rerr_packet.is_rrep = False
            rerr_packet.is_rerr = True
            rerr_packet.hop_count = 0
            rerr_packet.ttl = 64  # Initial TTL value for RERR
            self.packets.append(rerr_packet)
            self.env.process(self.transmit(rerr_packet))
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'forwarded RERR to precursor', precursor)
    
    def receive(self, pipe):
        while True:
            packet = yield pipe.get()
            if packet.sensedByN[self.nodeid] and not packet.collidedAtN[self.nodeid] and packet.onAirToN[self.nodeid]:  # start of reception
                if not self.isTransmitting:
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'started receiving packet', packet.seq, 'from', packet.txNodeId)
                    packet.onAirToN[self.nodeid] = False
                    self.totalEnergyConsumedJ += (self.conf.Rx_Powr * (packet.timeOnAir / 1000.0))  # Power (W) * time (s) = energy (J)  
                    self.isReceiving.append(True)
                else:  # if you were currently transmitting, you could not have sensed it
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'was transmitting, so could not receive packet', packet.seq)
                    packet.sensedByN[self.nodeid] = False
                    packet.onAirToN[self.nodeid] = False
            elif packet.sensedByN[self.nodeid]:  # end of reception
                try:
                    self.isReceiving[self.isReceiving.index(True)] = False
                except Exception:
                    pass
                self.airUtilization += packet.timeOnAir
                if packet.collidedAtN[self.nodeid]:
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'could not decode packet.')
                    continue
                packet.receivedAtN[self.nodeid] = True
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received packet', packet.seq, 'with delay', round(self.env.now - packet.genTime, 2))
                self.delays.append(self.env.now - packet.genTime)
                if packet.is_rreq:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is handling RREQ from', packet.origTxNodeId, 'to', packet.destId, 'RREQ_ID', packet.rreq_id)
                        self.handle_rreq(packet)
                elif packet.is_rrep:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is handling RREP from', packet.origTxNodeId, 'to', packet.destId, 'RREQ_ID', packet.rreq_id)
                        self.handle_rrep(packet)
                elif packet.is_rerr:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is handling RERR for', packet.destId)
                        self.handle_rerr(packet)
                elif packet.is_sdn_update:
                        self.handle_sdn_update(packet)
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'processed SDN update packet', packet.seq, 'from', packet.origTxNodeId)
                elif packet.destId == self.nodeid or packet.destId == NODENUM_BROADCAST:
                    if not packet.isAck and packet.wantAck and packet.destId == self.nodeid:
                        self.messageSeq["val"] += 1
                        messageSeq = self.messageSeq["val"]
                        self.messages.append(MeshMessage(self.nodeid, packet.origTxNodeId, self.env.now, messageSeq))
                        self.update_routing_table(packet.origTxNodeId, packet.txNodeId, packet.hop_count + 1, packet.seq)
                        ack_packet = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, packet.origTxNodeId, self.nodeid, 10, messageSeq, self.env.now, False, True, packet.seq, self.env.now, self.verboseprint)
                        ack_packet.next_hop = self.routing_table.get(packet.origTxNodeId).nextHop if packet.origTxNodeId in self.routing_table else None
                        if ack_packet.next_hop is None:
                            self.verboseprint(
                                "[ACK SEND NEXT_HOP_NONE]",
                                "time", round(self.env.now, 3),
                                "| node", self.nodeid,
                                "| ack_seq", ack_packet.seq,
                                "| replying_to", packet.origTxNodeId,
                                "| data_seq", packet.seq,
                                "| reverse_route_exists", (packet.origTxNodeId in self.routing_table),
                                "| reverse_valid", (self.routing_table[packet.origTxNodeId].valid if packet.origTxNodeId in self.routing_table else None),
                                "| reverse_nextHop", (self.routing_table[packet.origTxNodeId].nextHop if packet.origTxNodeId in self.routing_table else None),
                                "| hint", "No reverse route => ACK may get stuck",
                            )

                        ack_packet.hopLimit = 10
                        self.packets.append(ack_packet)
                        self.env.process(self.transmit(ack_packet))
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'sent ACK for packet', packet.seq, 'to', packet.origTxNodeId)
                    self.verboseprint('[DATA RECV] At time', round(self.env.now, 3), 'node', self.nodeid, 'received packet', packet.seq, 'from', packet.origTxNodeId)
                    if not packet.isAck:
                        orginTxNodeId = packet.origTxNodeId
                        for n in self.nodes:
                            if n.nodeid == orginTxNodeId:
                                orginTxNode = n
                                break
                        if orginTxNode.simRole == 'Sensor':
                            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is a Control node receiving a packet from Sensor node', orginTxNodeId)
                            if not packet.seq in self.SensorPacketsReceived.keys():
                                self.SensorPacketsReceived[packet.seq] = 0
                                if not packet.origTxNodeId in self.SensorPacketsReceivedOrigId.keys():
                                    self.SensorPacketsReceivedOrigId[packet.origTxNodeId] = {}
                                self.SensorPacketsReceivedOrigId[packet.origTxNodeId][packet.seq] = 0
                                if not packet.origTxNodeId in self.SensorPacketsDelays.keys():
                                    self.SensorPacketsDelays[packet.origTxNodeId] = []
                                self.SensorPacketsDelays[packet.origTxNodeId].append(self.env.now - packet.genTime)
                            self.SensorPacketsReceived[packet.seq] += 1
                            self.SensorPacketsReceivedOrigId[packet.origTxNodeId][packet.seq] += 1
                        
                        elif orginTxNode.simRole == "Control_Center":
                            if not packet.seq in self.BroadcastPacketsReceived.keys():
                                self.BroadcastPacketsReceived[packet.seq] = 0
                                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received broadcast packet', packet.seq)
                                pNew = MeshPacket_AODV(self.conf, self.nodes, packet.origTxNodeId, packet.destId, self.nodeid, packet.packetLen, packet.seq, packet.genTime, packet.wantAck, packet.isAck, packet.rreq_id, self.env.now, self.verboseprint)
                                pNew.hopLimit = packet.hopLimit - 1
                                pNew.next_hop = None
                                pNew.hop_count = packet.hop_count
                                pNew.ttl = packet.ttl
                                pNew.is_rreq = packet.is_rreq
                                pNew.is_rrep = packet.is_rrep
                                pNew.is_rerr = packet.is_rerr
                                self.packets.append(pNew)
                                self.env.process(self.transmit(pNew))
                                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasted broadcast packet', pNew.seq)
                                if not packet.origTxNodeId in self.BroadcastPacketsDelays.keys():
                                    self.BroadcastPacketsDelays[packet.origTxNodeId] = []
                                self.BroadcastPacketsDelays[packet.origTxNodeId].append(self.env.now - packet.genTime)
                            self.BroadcastPacketsReceived[packet.seq] += 1
                        
                        elif orginTxNode.simRole == "DM":
                            if not packet.seq in self.DMPacketsReceived.keys():
                                self.DMPacketsReceived[packet.seq] = 0
                                if not packet.origTxNodeId in self.DMPacketsReceivedOrigId.keys():
                                    self.DMPacketsReceivedOrigId[packet.origTxNodeId] = {}
                                self.DMPacketsReceivedOrigId[packet.origTxNodeId][packet.seq] = 0
                                if not packet.origTxNodeId in self.DMPacketsDelays.keys():
                                    self.DMPacketsDelays[packet.origTxNodeId] = []
                                self.DMPacketsDelays[packet.origTxNodeId].append(self.env.now - packet.genTime)
                            self.DMPacketsReceived[packet.seq] += 1
                            self.DMPacketsReceivedOrigId[packet.origTxNodeId][packet.seq] += 1
                    else:
                        if self.simRole == "Sensor":
                            if not packet.seq in self.SensorPacketsAcked.keys():
                                self.SensorPacketsAcked[packet.seq] = 0
                                self.ACKPacketsDelays.append(self.env.now - packet.genTime)
                            self.SensorPacketsAcked[packet.seq] += 1
                        elif self.simRole == "DM":
                            if not packet.seq in self.DMPacketsAcked.keys():
                                self.DMPacketsAcked[packet.seq] = 0
                                self.ACKPacketsDelays.append(self.env.now - packet.genTime)
                            self.DMPacketsAcked[packet.seq] += 1
                    # if packet.destId == NODENUM_BROADCAST and not self.isClientMute:
                    #         self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received broadcast packet', packet.seq)
                    #         pNew = MeshPacket_AODV(self.conf, self.nodes, packet.origTxNodeId, packet.destId, self.nodeid, packet.packetLen, packet.seq, packet.genTime, packet.wantAck, packet.isAck, packet.rreq_id, self.env.now, self.verboseprint)
                    #         pNew.hopLimit = packet.hopLimit - 1
                    #         pNew.next_hop = None
                    #         pNew.hop_count = packet.hop_count
                    #         pNew.ttl = packet.ttl
                    #         pNew.is_rreq = packet.is_rreq
                    #         pNew.is_rrep = packet.is_rrep
                    #         pNew.is_rerr = packet.is_rerr
                    #         self.packets.append(pNew)
                    #         self.env.process(self.transmit(pNew))
                    #         self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasted broadcast packet', pNew.seq)
                else:
                    if packet.hopLimit >= 0:
                        if not self.isClientMute and packet.next_hop == self.nodeid:
                            next_hop = self.routing_table.get(packet.destId).nextHop if packet.destId in self.routing_table else None
                            if packet.destId != NODENUM_BROADCAST and next_hop is None:
                                self.verboseprint(
                                    "[DATA FWD NEXT_HOP_NONE]",
                                    "time", round(self.env.now, 3),
                                    "| node", self.nodeid,
                                    "| seq", packet.seq,
                                    "| orig", packet.origTxNodeId,
                                    "| rx_from", packet.txNodeId,
                                    "| dest", packet.destId,
                                    "| pkt_next_hop_in", getattr(packet, "next_hop", None),
                                    "| route_exists", (packet.destId in self.routing_table),
                                    "| route_valid", (self.routing_table[packet.destId].valid if packet.destId in self.routing_table else None),
                                    "| route_nextHop", (self.routing_table[packet.destId].nextHop if packet.destId in self.routing_table else None),
                                    "| hopLimit_in", getattr(packet, "hopLimit", None),
                                    "| hop_count_in", getattr(packet, "hop_count", None),
                                    "| hint", "Forwarding requested but route table has no nextHop",
                                )

                            self.verboseprint('[DATA FWD] At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received packet', packet.seq, 'next hop', next_hop)
                            
                            pNew = MeshPacket_AODV(self.conf, self.nodes, packet.origTxNodeId, packet.destId, self.nodeid, packet.packetLen, packet.seq, packet.genTime, packet.wantAck, packet.isAck, packet.rreq_id, self.env.now, self.verboseprint)
                            pNew.hopLimit = packet.hopLimit - 1
                            pNew.next_hop = next_hop
                            pNew.hop_count = packet.hop_count +1
                            pNew.ttl = packet.ttl
                            pNew.is_rreq = packet.is_rreq
                            pNew.is_rrep = packet.is_rrep
                            pNew.is_rerr = packet.is_rerr
                            pNew.data = packet.data
                            pNew.is_sdn_update = packet.is_sdn_update
                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasted packet', pNew.seq, 'to', pNew.destId)
                        elif packet.destId == NODENUM_BROADCAST and not self.isClientMute:
                            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received broadcast packet', packet.seq)
                            pNew = MeshPacket_AODV(self.conf, self.nodes, packet.origTxNodeId, packet.destId, self.nodeid, packet.packetLen, packet.seq, packet.genTime, packet.wantAck, packet.isAck, packet.rreq_id, self.env.now, self.verboseprint)
                            pNew.hopLimit = packet.hopLimit - 1
                            pNew.next_hop = None
                            pNew.hop_count = packet.hop_count
                            pNew.ttl = packet.ttl
                            pNew.is_rreq = packet.is_rreq
                            pNew.is_rrep = packet.is_rrep
                            pNew.is_rerr = packet.is_rerr
                            pNew.data = packet.data
                            pNew.is_sdn_update = packet.is_sdn_update
                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasted broadcast packet', pNew.seq)
                    else:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'dropped packet', packet.seq, 'due to hop limit reached')
                for sentPacket in self.packets:
                    # check if ACK for message you currently have in queue
                    if sentPacket.txNodeId == self.nodeid and sentPacket.seq == packet.seq:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received implicit ACK for message in queue.')
                        ackReceived = True
                        sentPacket.ackReceived = True
                    # check if real ACK for message sent
                    if sentPacket.origTxNodeId == self.nodeid and packet.isAck and sentPacket.seq == packet.requestId:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received real ACK.')
                        realAckReceived = True
                        sentPacket.ackReceived = True
    def get_route_table(self):
        route_info = {}
        for destId, entry in self.routing_table.items():
            route_info[destId] = {
                'nextHop': entry.nextHop,
                'hopCount': entry.hopCount,
                'destSeqNum': entry.destSeqNum,
                'valid': entry.valid,
                'lifeTime': entry.lifeTime
            }
        return route_info
    
    def update_routing_table(self, destId, nextHop, hopCount, destSeqNum, valid=True,precursorList=[], lifeTime = 4000000):
        if nextHop is None:
            self.verboseprint(
                "[ROUTE INSTALL NEXT_HOP_NONE]",
                "time", round(self.env.now, 3),
                "| node", self.nodeid,
                "| dest", destId,
                "| hopCount", hopCount,
                "| destSeqNum", destSeqNum,
                "| hint", "You are installing a broken route entry",
            )

        self.routing_table[destId] = RouteEntry(
            destId=destId,
            nextHop=nextHop,
            hopCount=hopCount,
            destSeqNum=destSeqNum,
            valid=valid,
            precursorList=precursorList,
            lifeTime=self.env.now + lifeTime  
        )
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'updated routing table for', destId, 'via', nextHop)

    def handle_sdn_update(self,packet):
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'handling SDN update packet', packet.seq)
        # Process the SDN update contained in the packet
        # This is a placeholder for actual SDN update logic
        # For example, updating flow tables or routing policies based on the packet content
        pass

class RouteEntry:
    def __init__(self, destId, nextHop, hopCount, destSeqNum, valid, precursorList, lifeTime):
        self.destId = destId
        self.nextHop = nextHop
        self.hopCount = hopCount
        self.destSeqNum = destSeqNum
        self.valid = valid
        self.precursorList = precursorList  # list of nodeIds
        self.lifeTime = lifeTime  # expiration time
        