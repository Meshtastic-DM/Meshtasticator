from    lib.node import MeshNode
from    lib.packet_aodv import MeshPacket_AODV
from    lib.packet import NODENUM_BROADCAST, MeshMessage
import  simpy
import  random

class MeshNode_AODV(MeshNode):
    """
    This class extends MeshNode to implement AODV routing protocol functionality.
    It overrides necessary methods and adds new attributes and methods specific to AODV.
    """
    def __init__(self, conf, nodes, env, bc_pipe, nodeid, period, messages, packetsAtN, packets, delays, nodeConfig, messageSeq, verboseprint):
        super().__init__( conf, nodes, env, bc_pipe, nodeid, period, messages, packetsAtN, packets, delays, nodeConfig, messageSeq, verboseprint)
        self.routing_table = {}  # key: destination nodeId, value: next hop nodeId
        self.rreq_id_counter = 0  # Counter for generating unique RREQ IDs
        self.pending_rreq = {}  # key: (destId, rreq_id), value: list of packets waiting for route
        self.seq_num = 0  # Sequence number for this node
        #self.messages = []  # List to store generated messages
        self.routing_table: dict[int, RouteEntry] = {}
        self.transmitter = simpy.Resource(env, 1)
        self.conf.SELECTED_ROUTER_TYPE = self.conf.ROUTER_TYPE.AODV
        self.processed_rreq = set()  # Set to track processed RREQs to avoid loops
        self.processed_rrep = set()  # Set to track processed RREPs to avoid loops
        self.forwarded_rrep = set()  # Set to track forwarded RREPs to avoid loops

    def send_packet(self, destId, wantAck=False):
        plen = 20
        self.seq_num += 1
        self.messageSeq["val"] += 1
        messageSeq = self.messageSeq["val"]
        self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))
        p = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, destId, self.nodeid, plen, messageSeq, self.env.now, wantAck, False, None, self.env.now, self.verboseprint)
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'preparing to send packet', p.seq, 'to', destId)
        if destId != NODENUM_BROADCAST:
            if destId in self.routing_table and self.routing_table[destId].valid:
                pNew = MeshPacket_AODV(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, p.isAck, None, self.env.now, self.verboseprint)
                pNew.hopLimit = p.hopLimit - 1
                pNew.next_hop = self.routing_table[destId].nextHop if destId in self.routing_table else None
                self.packets.append(pNew)
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is sending packet', pNew.seq, 'to', pNew.destId, 'via next hop', self.routing_table[destId].nextHop)
                self.env.process(self.transmit(pNew))
            else:
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'no valid route to', destId, 'initiating route discovery')
                # Initiate route discovery
                self.initiate_route_discovery(destId)
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is initiating route discovery for', destId)
                # Store the packet to be sent once the route is discovered
                if (destId, self.rreq_id_counter) not in self.pending_rreq:
                    self.pending_rreq[(destId, self.rreq_id_counter)] = []
                self.pending_rreq[(destId, self.rreq_id_counter)].append(p)
        else:
            # Broadcast packet
            pNew = MeshPacket_AODV(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, p.isAck, None, self.env.now, self.verboseprint)
            pNew.hopLimit = p.hopLimit - 1
            self.packets.append(pNew)
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
        rreq_packet = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, destId, self.nodeid, 10, self.seq_num, self.env.now, False, False, rreq_id, self.env.now, self.verboseprint,rreq_id=rreq_id)
        rreq_packet.is_rreq = True
        rreq_packet.is_rrep = False
        rreq_packet.is_rerr = False
        rreq_packet.hop_count = 0
        rreq_packet.ttl = 64  # Initial TTL value for RREQ
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'created RREQ for', destId, 'with RREQ_ID', rreq_id)
        self.packets.append(rreq_packet)
        self.env.process(self.transmit(rreq_packet))
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'transmitted RREQ for', destId, 'with RREQ_ID', rreq_id)

    def handle_rreq(self, packet):
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received RREQ from', packet.origTxNodeId, 'for', packet.destId, 'RREQ_ID', packet.rreq_id)
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
            self.routing_table[packet.origTxNodeId] = RouteEntry(
                destId=packet.origTxNodeId,
                nextHop=packet.txNodeId,
                hopCount=packet.hop_count + 1,
                destSeqNum=packet.seq,
                valid=True,
                precursorList=[],
                lifeTime=self.env.now + 300000  
            )
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'updated reverse route to', packet.origTxNodeId, 'via', packet.txNodeId)
        # If this node is the destination, send RREP    
        if packet.destId == self.nodeid:
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is destination for RREQ, sending RREP to', packet.origTxNodeId)
            self.send_rrep(packet)
        elif packet.ttl > 1:
            # Forward the RREQ
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'forwarding RREQ from', packet.origTxNodeId, 'for', packet.destId, 'RREQ_ID', packet.rreq_id)
            packet.hop_count += 1
            packet.ttl -= 1
            packet.txNodeId = self.nodeid
            self.packets.append(packet)
            self.env.process(self.transmit(packet)) # Rebroadcast the RREQ
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
        rrep_packet = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, rreq_packet.origTxNodeId, self.nodeid, 10, self.seq_num, self.env.now, False, False, None, self.env.now, self.verboseprint)
        rrep_packet.is_rreq = False
        rrep_packet.is_rrep = True
        rrep_packet.is_rerr = False
        rrep_packet.hop_count = 0
        rrep_packet.ttl = 64  # Initial TTL value for RREP
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
        self.routing_table[packet.origTxNodeId] = RouteEntry(
            destId=packet.origTxNodeId,
            nextHop=packet.txNodeId,
            hopCount=1,
            destSeqNum=packet.seq,
            valid=True,
            precursorList=[],
            lifeTime=self.env.now + 3000000  # Example lifetime
        )
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'updated forward route to', packet.origTxNodeId, 'via', packet.txNodeId)
        # If this node is the source of the RREQ, send pending packets
        if packet.destId == self.nodeid:
            key = (packet.origTxNodeId, packet.rreq_id)
            if key in self.pending_rreq:
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'sending pending packets for', packet.origTxNodeId)
                for p in self.pending_rreq[key]:
                    pNew = MeshPacket_AODV(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, p.isAck, None, self.env.now, self.verboseprint)
                    self.packets.append(pNew)
                    self.env.process(self.transmit(pNew))
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'transmitted pending packet', pNew.seq, 'to', pNew.destId)
                del self.pending_rreq[key]
        elif packet.ttl > 1:
            # Forward the RREP
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'forwarding RREP for', packet.origTxNodeId)
            packet.hop_count += 1
            packet.ttl -= 1
            packet.txNodeId = self.nodeid
            self.packets.append(packet)
            self.env.process(self.transmit(packet)) # Rebroadcast the RREP
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasted RREP for', packet.origTxNodeId)

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

            if packet.txNodeId == self.nodeid:
                continue  # Ignore packets sent by self
            if not packet.sensedByN[self.nodeid]:
                continue  # Ignore packets not sensed
            if not packet.collidedAtN[self.nodeid]:
                packet.receivedAtN[self.nodeid] = True
                self.usefulPackets += 1
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received packet', packet.seq, 'from', packet.origTxNodeId)
                if packet.is_rreq:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is handling RREQ from', packet.origTxNodeId, 'for', packet.destId, 'RREQ_ID', packet.rreq_id)
                        self.handle_rreq(packet)
                elif packet.is_rrep:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is handling RREP from', packet.txNodeId, 'for', packet.origTxNodeId)
                        self.handle_rrep(packet)
                elif packet.is_rerr:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is handling RERR for', packet.destId)
                        self.handle_rerr(packet)
                elif packet.destId == self.nodeid or packet.destId == NODENUM_BROADCAST:
                    if not packet.isAck and packet.wantAck:
                        ack_packet = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, packet.origTxNodeId, self.nodeid, 10, packet.seq, self.env.now, False, True, None, self.env.now, self.verboseprint)
                        self.packets.append(ack_packet)
                        self.env.process(self.transmit(ack_packet))
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'sent ACK for packet', packet.seq, 'to', packet.origTxNodeId)
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received packet', packet.seq, 'from', packet.origTxNodeId)
                else:
                    if packet.hopLimit > 1:
                        if not self.isClientMute and packet.next_hop == self.nodeid:
                            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received packet', packet.seq)
                            next_hop = self.routing_table.get(packet.destId).nextHop if packet.destId else None
                            pNew = MeshPacket_AODV(self.conf, self.nodes, packet.origTxNodeId, packet.destId, self.nodeid, packet.packetLen, packet.seq, packet.genTime, packet.wantAck, packet.isAck, packet.rreq_id, self.env.now, self.verboseprint)
                            pNew.hopLimit = packet.hopLimit - 1
                            pNew.next_hop = next_hop
                            pNew.hop_count = packet.hop_count
                            pNew.ttl = packet.ttl
                            pNew.is_rreq = packet.is_rreq
                            pNew.is_rrep = packet.is_rrep
                            pNew.is_rerr = packet.is_rerr
                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasted packet', pNew.seq, 'to', pNew.destId)
                    else:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'dropped packet', packet.seq, 'due to hop limit reached')
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

class RouteEntry:
    def __init__(self, destId, nextHop, hopCount, destSeqNum, valid, precursorList, lifeTime):
        self.destId = destId
        self.nextHop = nextHop
        self.hopCount = hopCount
        self.destSeqNum = destSeqNum
        self.valid = valid
        self.precursorList = precursorList  # list of nodeIds
        self.lifeTime = lifeTime  # expiration time
        