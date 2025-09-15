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
    def __init__(self, env, conf, nodeid, x, y, z, isRepeater=False):
        super().__init__(env, conf, nodeid, x, y, z, isRepeater)
        self.routing_table = {}  # key: destination nodeId, value: next hop nodeId
        self.rreq_id_counter = 0  # Counter for generating unique RREQ IDs
        self.pending_rreq = {}  # key: (destId, rreq_id), value: list of packets waiting for route
        self.seq_num = 0  # Sequence number for this node
        self.messages = []  # List to store generated messages
        self.routing_table: dict[int, RouteEntry] = {}
        self.transmitter = simpy.Resource(env, 1)
        self.conf.SELECTED_ROUTER_TYPE = self.conf.ROUTER_TYPE.AODV

    def send_packet(self, destId, plen, wantAck=False):
        self.seq_num += 1
        messageSeq = self.seq_num
        p = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, destId, self.nodeid, plen, messageSeq, self.env.now, wantAck, False, None, self.env.now, self.verboseprint)
        if destId != NODENUM_BROADCAST:
            if destId in self.routing_table and self.routing_table[destId].valid:
                pNew = MeshPacket_AODV(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, p.isAck, None, self.env.now, self.verboseprint)
                pNew.hopLimit = p.hopLimit - 1
                self.packets.append(pNew)
                self.env.process(self.transmit(pNew))
            else:
                # Initiate route discovery
                self.initiate_route_discovery(destId)
                # Store the packet to be sent once the route is discovered
                if (destId, self.rreq_id_counter) not in self.pending_rreq:
                    self.pending_rreq[(destId, self.rreq_id_counter)] = []
                self.pending_rreq[(destId, self.rreq_id_counter)].append(p)
        else:
            # Broadcast packet
            pNew = MeshPacket_AODV(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, p.isAck, None, self.env.now, self.verboseprint)
            pNew.hopLimit = p.hopLimit - 1
            self.packets.append(pNew)
            self.env.process(self.transmit(pNew))
        self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))

    def initiate_route_discovery(self, destId):
        self.rreq_id_counter += 1
        rreq_id = self.rreq_id_counter
        rreq_packet = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, destId, self.nodeid, 20, self.seq_num, self.env.now, False, False, rreq_id, self.env.now, self.verboseprint)
        rreq_packet.is_rreq = True
        rreq_packet.is_rrep = False
        rreq_packet.is_rerr = False
        rreq_packet.hop_count = 0
        rreq_packet.ttl = 64  # Initial TTL value for RREQ
        self.packets.append(rreq_packet)
        self.env.process(self.transmit(rreq_packet))

    def handle_rreq(self, packet):
        if packet.rreq_id is None:
            return  # Not a valid RREQ packet

        # Check if this RREQ has been processed before
        if (packet.origTxNodeId, packet.rreq_id) in self.processed_rreq:
            return  # Already processed

        # Mark this RREQ as processed
        self.processed_rreq.add((packet.origTxNodeId, packet.rreq_id))

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
                lifeTime=self.env.now + 30000  # Example lifetime
            )
        # If this node is the destination, send RREP    
        if packet.destId == self.nodeid:
            self.send_rrep(packet)
        elif packet.ttl > 1:
            # Forward the RREQ
            packet.hop_count += 1
            packet.ttl -= 1
            packet.txNodeId = self.nodeid
            self.packets.append(packet)
            self.env.process(self.transmit(packet)) # Rebroadcast the RREQ
    def send_rrep(self, rreq_packet):
        rrep_packet = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, rreq_packet.origTxNodeId, self.nodeid, 20, self.seq_num, self.env.now, False, False, None, self.env.now, self.verboseprint)
        rrep_packet.is_rreq = False
        rrep_packet.is_rrep = True
        rrep_packet.is_rerr = False
        rrep_packet.hop_count = 0
        rrep_packet.ttl = 64  # Initial TTL value for RREP
        # Update routing table with forward route to the destination
        self.routing_table[rreq_packet.origTxNodeId] = RouteEntry(
            destId=rreq_packet.origTxNodeId,
            nextHop=rreq_packet.txNodeId,
            hopCount=1,
            destSeqNum=rreq_packet.seq,
            valid=True,
            precursorList=[],
            lifeTime=self.env.now + 30000  # Example lifetime
        )
        self.packets.append(rrep_packet)
        self.env.process(self.transmit(rrep_packet))

    def handle_rrep(self, packet):
        if not packet.is_rrep:
            return  # Not a valid RREP packet

        # Update routing table with forward route to the destination
        self.routing_table[packet.origTxNodeId] = RouteEntry(
            destId=packet.origTxNodeId,
            nextHop=packet.txNodeId,
            hopCount=1,
            destSeqNum=packet.seq,
            valid=True,
            precursorList=[],
            lifeTime=self.env.now + 30000  # Example lifetime
        )
        # If this node is the source of the RREQ, send pending packets
        if packet.destId == self.nodeid:
            key = (packet.origTxNodeId, packet.rreq_id)
            if key in self.pending_rreq:
                for p in self.pending_rreq[key]:
                    pNew = MeshPacket_AODV(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, p.isAck, None, self.env.now, self.verboseprint)
                    pNew.hopLimit = p.hopLimit - 1
                    self.packets.append(pNew)
                    self.env.process(self.transmit(pNew))
                del self.pending_rreq[key]
        elif packet.ttl > 1:
            # Forward the RREP
            packet.hop_count += 1
            packet.ttl -= 1
            packet.txNodeId = self.nodeid
            self.packets.append(packet)
            self.env.process(self.transmit(packet)) # Rebroadcast the RREP

    def handle_rerr(self, packet):
        if not packet.is_rerr:
            return  # Not a valid RERR packet
        # Invalidate the route to the unreachable destination
        if packet.destId in self.routing_table:
            self.routing_table[packet.destId].valid = False
        # Forward the RERR to precursors if any
        for precursor in self.routing_table.get(packet.destId, RouteEntry(None, None, None, None, False, [], None)).precursorList:
            rerr_packet = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, precursor, self.nodeid, 20, self.seq_num, self.env.now, False, False, None, self.env.now, self.verboseprint)
            rerr_packet.is_rreq = False
            rerr_packet.is_rrep = False
            rerr_packet.is_rerr = True
            rerr_packet.hop_count = 0
            rerr_packet.ttl = 64  # Initial TTL value for RERR
            self.packets.append(rerr_packet)
            self.env.process(self.transmit(rerr_packet))
    
    def receive(self, pipe):
        while True:
            packet = yield pipe.get()
            if packet.destId == self.nodeid or packet.destId == NODENUM_BROADCAST:
                if packet.is_rreq:
                    self.handle_rreq(packet)
                elif packet.is_rrep:
                    self.handle_rrep(packet)
                elif packet.is_rerr:
                    self.handle_rerr(packet)
                else:
                    if not packet.isAck and packet.wantAck:
                        ack_packet = MeshPacket_AODV(self.conf, self.nodes, self.nodeid, packet.origTxNodeId, self.nodeid, 10, packet.seq, self.env.now, False, True, None, self.env.now, self.verboseprint)
                        self.packets.append(ack_packet)
                        self.env.process(self.transmit(ack_packet))
                    if not packet.collidedAtN[self.nodeid]:
                        packet.receivedAtN[self.nodeid] = True
                        self.usefulPackets += 1
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received packet', packet.seq, 'from', packet.origTxNodeId)
            else:
                if packet.hopLimit > 1:
                    if not self.isClientMute:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received packet', packet.seq)
                        pNew = MeshPacket_AODV(self.conf, self.nodes, packet.origTxNodeId, packet.destId, self.nodeid, packet.packetLen, packet.seq, packet.genTime, packet.wantAck, packet.isAck, packet.rreq_id, self.env.now, self.verboseprint)
                        pNew.hopLimit = packet.hopLimit - 1
                        pNew.hop_count = packet.hop_count
                        pNew.ttl = packet.ttl
                        pNew.is_rreq = packet.is_rreq
                        pNew.is_rrep = packet.is_rrep
                        pNew.is_rerr = packet.is_rerr
                        self.packets.append(pNew)
                        self.env.process(self.transmit(pNew))
                else:
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'dropped packet', packet.seq, 'due to hop limit reached')


class RouteEntry:
    def __init__(self, destId, nextHop, hopCount, destSeqNum, valid, precursorList, lifeTime):
        self.destId = destId
        self.nextHop = nextHop
        self.hopCount = hopCount
        self.destSeqNum = destSeqNum
        self.valid = valid
        self.precursorList = precursorList  # list of nodeIds
        self.lifeTime = lifeTime  # expiration time
        