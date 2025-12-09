from lib.packet import MeshPacket
from lib.phy import airtime


class MeshPacket_ZRP(MeshPacket):
    """
    ZRP packet supporting:
      - IARP : Intrazone proactive routing updates
      - IERP : Interzone route discovery (RREQ / RREP)
      - DATA : packet_type=None (normal data / ACK)

    IERP packets:
      - packet_type = "IERP"
      - ierp_type   ∈ {"RREQ", "RREP"}
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
        packet_type=None,          # "IARP", "IERP" or None (DATA)

        # IARP fields
        iarp_seq_num=None,

        # IERP fields
        ierp_type=None,            # "RREQ" or "RREP"
        ierp_id=None,
        ierp_destId=None,

        # Common ZRP fields
        hop_count=0,
        covered_nodes=None,        # list of node IDs already covered by this RREQ
    ):
        # Initialize base MeshPacket (PHY + generic fields)
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
        self.packet_type = packet_type
        self.zone_radius = getattr(conf, "ZRP_ZONE_RADIUS", 2)
        self.hop_count = hop_count

        # -----------------------------
        # IARP Fields
        # -----------------------------
        self.iarp_seq_num = iarp_seq_num

        # -----------------------------
        # IERP Fields
        # -----------------------------
        self.ierp_type = ierp_type
        self.ierp_id = ierp_id
        self.ierp_destId = ierp_destId

        # Route accumulation (for RREP; not yet fully used)
        self.route_path = []

        # Duplicate / bordercast resolution set
        # Keep what caller passed, don't overwrite it
        self.covered_nodes = list(covered_nodes) if covered_nodes is not None else []

        # -----------------------------
        # Packet length normalisation
        # -----------------------------
        # Only override length for control packets.
        # DATA packets (packet_type=None) keep the caller's packetLen.
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
