from lib.packet import MeshPacket

class MeshPacket_AODV(MeshPacket):
    """
    The new class was created to simulate a packet used in AODV routing protocol.
    This class inherits from MeshPacket and adds attributes specific to AODV.
    """
    def __init__(self, conf, nodes, origTxNodeId, destId, txNodeId, plen, seq, genTime, 
                 wantAck, isAck, requestId, now, verboseprint,
                 # New parameters for AODV functionality
                 hop_count=0, ttl=64, rreq_id=None):
        
        # Call the parent class constructor
        super().__init__(conf, nodes, origTxNodeId, destId, txNodeId, plen, seq, 
                        genTime, wantAck, isAck, requestId, now, verboseprint)
        
        # Add new attributes specific to AODV
        self.hop_count = hop_count  # Number of hops the packet has traversed
        self.ttl = ttl  # Time to live for the packet
        self.rreq_id = rreq_id  # Unique ID for RREQ packets
        
        # Additional tracking attributes
        self.route_discovery_time = None  # Time taken for route discovery
        self.is_rrep = False  # Flag to indicate if this is a RREP packet
        self.is_rerr = False  # Flag to indicate if this is a RERR packet

        self.next_hop = None  # Next hop nodeId for the packet
        
    def increment_hop_count(self):
        """Increment the hop count by one"""
        self.hop_count += 1
    
    def decrement_ttl(self):
        """Decrement the TTL by one"""
        if self.ttl > 0:
            self.ttl -= 1
    
    def is_ttl_expired(self):
        """Check if the TTL has expired"""
        return self.ttl <= 0