from lib.packet import MeshPacket
from lib.phy import airtime


class MeshPacket_ZRP(MeshPacket):
    """
    ZRP control packet supporting two types:
        - IARP : Intrazone proactive routing updates
        - IERP : Interzone route discovery (RREQ / RREP)

    IERP packets specify:
        ierp_type ∈ {"RREQ", "RREP"}
    """

    def __init__(
        self,
        conf,
        nodes,
        origTxNodeId,
        destId,
        txNodeId,
        packetLen,
        seq,
        genTime,
        wantAck,
        isAck,
        requestId,
        txTime,
        verboseprint,

        # ZRP packet type
        packet_type=None,          # "IARP" or "IERP"

        # IARP fields
        iarp_seq_num=None,

        # IERP fields
        ierp_type=None,       # "RREQ" or "RREP"
        ierp_id=None,

        # Common ZRP fields
        hop_count=0,          # <— INCLUDED
    ):
        # Initialize PHY-layer MeshPacket
        super().__init__(
            conf,
            nodes,
            origTxNodeId,
            destId,
            txNodeId,
            packetLen,
            seq,
            genTime,
            wantAck,
            isAck,
            requestId,
            txTime,
            verboseprint,
        )

        # -----------------------------
        # ZRP Common Metadata
        # -----------------------------
        self.packet_type = packet_type          # "IARP" or "IERP"
        self.zone_radius = getattr(conf, "ZRP_ZONE_RADIUS", 2)
        self.hop_count = hop_count              # <— REQUIRED FOR ZRP

        # -----------------------------
        # IARP Fields
        # -----------------------------
        self.iarp_seq_num = iarp_seq_num

        # -----------------------------
        # IERP Fields
        # -----------------------------
        self.ierp_type = ierp_type              # "RREQ" or "RREP"
        self.ierp_id = ierp_id

        # Route accumulation (for RREP)
        self.route_path = []

        # Duplicate suppression
        self.covered_nodes = []

        # -----------------------------
        # Packet Sizes
        # -----------------------------
        if self.packet_type == "IARP":
            self.packetLen = getattr(conf, "IARP_PACKET_LEN", 20)

        elif self.packet_type == "IERP":
            if self.ierp_type == "RREQ":
                self.packetLen = getattr(conf, "IERP_RREQ_PACKET_LEN", 28)
            elif self.ierp_type == "RREP":
                self.packetLen = getattr(conf, "IERP_RREP_PACKET_LEN", 26)
            else:
                self.packetLen = getattr(conf, "IERP_PACKET_LEN", 30)

        # -----------------------------
        # Recalculate airtime
        # -----------------------------
        self.timeOnAir = airtime(
            conf,
            conf.SFMODEM[conf.MODEM],
            conf.CRMODEM[conf.MODEM],
            self.packetLen,
            conf.BWMODEM[conf.MODEM],
        )
