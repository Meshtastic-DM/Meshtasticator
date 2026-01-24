from lib.node_aodv import MeshNode_AODV
import json
import os
from lib.common import calc_dist, find_random_position
from lib.mac import set_transmit_delay, get_retransmission_msec
from lib.packet_aodv import MeshPacket_AODV
from lib.phy import check_collision, is_channel_active, airtime
from lib.packet import NODENUM_BROADCAST
from lib.packet_aodv import MeshPacket_AODV
# cross-platform file locking: prefer fcntl (Unix), fall back to msvcrt (Windows),
# otherwise no-op (single-process or best-effort)
try:
    import fcntl

    def _lock(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)

    def _unlock(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
except Exception:
    try:
        import msvcrt

        def _lock(f):
            # Lock a single byte (Windows doesn't support flock). This is a best-effort
            # advisory lock to avoid concurrent writes when running multiple processes.
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            except Exception:
                # if locking fails, continue without blocking
                pass

        def _unlock(f):
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
    except Exception:
        # No locking available (e.g., exotic environment). Use no-op functions.
        def _lock(f):
            return

        def _unlock(f):
            return


class MeshNode_SDN(MeshNode_AODV):
    """
    Subclass of MeshNode_AODV that implements Software Defined Networking (SDN) capabilities.
    """
    def __init__(self, *args, **kwargs):
        super(MeshNode_SDN, self).__init__(*args, **kwargs)
        self.sdn_node_num = None  # Node number of the SDN controller
        self.sdn_node_hop_count = None  # Hop count to the SDN controller
        self.processed_sdn_update_packets = set()  # Track processed SDN update packets to avoid duplicates
       

    

    def sdn_reliable_retransmit(self, p):
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
                    "[SDN RETX EXIT ACK]",
                    "time", round(self.env.now, 3),
                    "| node", self.nodeid,
                    "| seq", p.seq,
                    "| destSeqNum", p.data.get("destSeqNum"),
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
                        "[SDN RETX EXIT NO ROUTE]",
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
                    "[SDN RETX SEND]",
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
                    "[SDN RETX EXIT FAIL]",
                    "time", round(self.env.now, 3),
                    "| node", self.nodeid,
                    "| seq", p.seq,
                    "| destSeqNum", p.data.get("destSeqNum"),
                )
                break

    def update_routing_table(self, destId, nextHop, hopCount, destSeqNum, valid=True, precursorList=None, lifeTime=300000):
        
        # Normalize precursorList to a serializable list
        pl = precursorList if precursorList is not None else []

        # Pass the normalized precursor list to the base implementation
        super().update_routing_table(destId, nextHop, hopCount, destSeqNum, valid, pl, lifeTime)

        if self.sdn_node_num is not None and self.simRole != 'sdn_node' and hopCount <=7 and destId != self.sdn_node_num:
            route_info_data = {
                'selfId': self.nodeid,
                'destId': destId,
                'nextHop': nextHop,
                'hopCount': hopCount,
                'destSeqNum': destSeqNum,
                'valid': valid,
                'precursorList': pl,
                'lifeTime': lifeTime
            }
            self.send_sdn_route_update(self.sdn_node_num, route_info_data)
        

    def send_sdn_route_update(self, controller_node_num, route_info_data):
        if controller_node_num is not None and self.simRole == "DM":
            p = self.send_packet(controller_node_num, data=route_info_data, is_sdn_update=True)
            self.verboseprint(
                "[SDN UPDATE SEND]",
                "time", round(self.env.now, 3),
                "| node", self.nodeid,
                "| to controller", controller_node_num,
                "| seq", p.seq,
                "| wantAck", p.wantAck,
                "| destSeqNum", route_info_data.get("destSeqNum"),
            )
            # Only start retransmission for unicast updates that want ACK
            if p.wantAck:
                self.env.process(self.sdn_reliable_retransmit(p))
                self.verboseprint(
                    "[SDN UPDATE RELIABLE START]",
                    "time", round(self.env.now, 3),
                    "| node", self.nodeid,
                    "| seq", p.seq,
                    "| maxRetx", getattr(self.conf, "maxRetransmission", None),
                )


    def handle_sdn_update(self, packet):
        self.verboseprint(
            "[SDN UPDATE RX]",
            "time", round(self.env.now, 3),
            "| node", self.nodeid,
            "| from", packet.origTxNodeId,
            "| seq", packet.seq,
            "| hop", packet.hop_count,
        )
        if packet.seq in self.processed_sdn_update_packets and packet.destId == NODENUM_BROADCAST:
            return  # Skip already processed packets
        if packet.seq in self.processed_sdn_update_packets and packet.destId == self.nodeid:
            return  # Skip already processed packets
        self.processed_sdn_update_packets.add(packet.seq)
        if self.simRole != 'sdn_node':
            if packet.destId == NODENUM_BROADCAST: #Nearest SDN node is configuring
                self.update_sdn_configuration(packet)
                if packet.hopLimit >0 and not self.isClientMute:
                    pNew = MeshPacket_AODV(self.conf, self.nodes, packet.origTxNodeId, packet.destId, self.nodeid, packet.packetLen, packet.seq, packet.genTime, packet.wantAck, packet.isAck, packet.rreq_id, self.env.now, self.verboseprint)
                    pNew.hopLimit = packet.hopLimit - 1
                    pNew.next_hop = None
                    pNew.hop_count = packet.hop_count +1
                    pNew.ttl = packet.ttl
                    pNew.is_rreq = packet.is_rreq
                    pNew.is_rrep = packet.is_rrep
                    pNew.is_rerr = packet.is_rerr
                    pNew.data = packet.data
                    pNew.is_sdn_update = packet.is_sdn_update
                    self.packets.append(pNew)
                    self.env.process(self.transmit(pNew))
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasted broadcast packet', pNew.seq)
            elif packet.destId != self.nodeid:
                if not self.isClientMute and self.nodeid == packet.next_hop:
                    # Forward the packet towards its destination
                    entry = self.routing_table.get(packet.destId)
                    next_hop = entry.nextHop if entry else None

                    if next_hop is None:
                        self.verboseprint(
                            "[SDN NH NONE]",
                            "time", round(self.env.now, 3),
                            "| node", self.nodeid,
                            "| seq", packet.seq,
                            "| dest", packet.destId,
                        )
                    pNew = MeshPacket_AODV(self.conf, self.nodes, packet.origTxNodeId, packet.destId, self.nodeid, packet.packetLen, packet.seq, packet.genTime, packet.wantAck, packet.isAck, packet.rreq_id, self.env.now, self.verboseprint)
                    pNew.hopLimit = packet.hopLimit -1
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
                    self.verboseprint(
                        "[SDN UPDATE FWD]",
                        "time", round(self.env.now, 3),
                        "| node", self.nodeid,
                        "| seq", pNew.seq,
                        "| dest", pNew.destId,
                        "| next_hop", next_hop,
                    )
        if self.simRole == 'sdn_node' and  (packet.origTxNodeId != self.nodeid):
            route_info = packet.data
            self.write_adajecny_data_into_json(route_info)
            self.verboseprint('At time', round(self.env.now, 3), 'controller node', self.nodeid, 'updated routing info from SDN packet', packet.seq, 'from', packet.origTxNodeId)
    
    def write_adajecny_data_into_json(self, route_info):
        
        json_file_path = 'sdn_route_info.json'
        
        with open(json_file_path, 'a+') as f:
            # Acquire exclusive lock (cross-platform helper)
            _lock(f)
            try:
                # Move to beginning to read existing data
                f.seek(0)
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = []
                except (json.JSONDecodeError, ValueError):
                    data = []

                # Append new route info
                data.append(route_info)

                # Write back to file
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)
            finally:
                # Release lock
                _unlock(f)

    def update_sdn_configuration(self, packet):
        self.verboseprint(
            "[SDN CFG RX]",
            "time", round(self.env.now, 3),
            "| node", self.nodeid,
            "| from", packet.origTxNodeId,
            "| hop", packet.hop_count,
            "| seq", packet.seq,
            "| current_ctrl", self.sdn_node_num,
            "| current_hop", self.sdn_node_hop_count,
        )

        updated = False
        reason = None

        if self.sdn_node_num is None:
            self.sdn_node_num = packet.origTxNodeId
            self.sdn_node_hop_count = packet.hop_count
            updated = True
            reason = "FIRST_SET"

            self.verboseprint(
                "[SDN CFG UPDATE]",
                "time", round(self.env.now, 3),
                "| node", self.nodeid,
                "| reason", reason,
                "| controller", self.sdn_node_num,
                "| hop", self.sdn_node_hop_count,
                "| seq", packet.seq,
            )

        else:
            # Better hop-count controller
            if packet.hop_count < self.sdn_node_hop_count:
                old_ctrl = self.sdn_node_num
                old_hop = self.sdn_node_hop_count

                self.sdn_node_num = packet.origTxNodeId
                self.sdn_node_hop_count = packet.hop_count
                updated = True
                reason = "BETTER_HOP"

                self.verboseprint(
                    "[SDN CFG UPDATE]",
                    "time", round(self.env.now, 3),
                    "| node", self.nodeid,
                    "| reason", reason,
                    "| old_ctrl", old_ctrl,
                    "| old_hop", old_hop,
                    "| new_ctrl", self.sdn_node_num,
                    "| new_hop", self.sdn_node_hop_count,
                    "| seq", packet.seq,
                )

            else:
                # Same controller, newer seq than current route entry -> update route info
                if packet.origTxNodeId == self.sdn_node_num:
                    entry = self.routing_table.get(self.sdn_node_num)
                    entry_seq = entry.destSeqNum if entry is not None else -1

                    if packet.seq > entry_seq:
                        updated = True
                        reason = "NEWER_SEQ"

                        self.verboseprint(
                            "[SDN CFG UPDATE]",
                            "time", round(self.env.now, 3),
                            "| node", self.nodeid,
                            "| reason", reason,
                            "| controller", self.sdn_node_num,
                            "| pkt_seq", packet.seq,
                            "| entry_seq", entry_seq,
                            "| hop", packet.hop_count,
                        )
                    else:
                        self.verboseprint(
                            "[SDN CFG IGNORE]",
                            "time", round(self.env.now, 3),
                            "| node", self.nodeid,
                            "| reason", "OLD_SEQ",
                            "| controller", self.sdn_node_num,
                            "| pkt_seq", packet.seq,
                            "| entry_seq", entry_seq,
                            "| hop", packet.hop_count,
                        )
                else:
                    self.verboseprint(
                        "[SDN CFG IGNORE]",
                        "time", round(self.env.now, 3),
                        "| node", self.nodeid,
                        "| reason", "DIFFERENT_CTRL",
                        "| pkt_ctrl", packet.origTxNodeId,
                        "| current_ctrl", self.sdn_node_num,
                        "| pkt_hop", packet.hop_count,
                        "| current_hop", self.sdn_node_hop_count,
                    )

        if updated:
            self.update_routing_table(
                self.sdn_node_num,
                packet.txNodeId,
                packet.hop_count + 1,
                packet.seq,
                valid=True
            )

            # log route install
            self.verboseprint(
                "[SDN CFG ROUTE SET]",
                "time", round(self.env.now, 3),
                "| node", self.nodeid,
                "| ctrl", self.sdn_node_num,
                "| nextHop", packet.txNodeId,
                "| hopCount", packet.hop_count + 1,
                "| seq", packet.seq,
                "| reason", reason,
            )

            '''for dest_id in self.routing_table.keys():
                if self.routing_table[dest_id].hopCount == 1:
                    route_info_data = {
                    'selfId': self.nodeid,
                    'destId': dest_id,
                    'nextHop': self.routing_table[dest_id].nextHop,
                    'hopCount':self.routing_table[dest_id].hopCount,
                    'destSeqNum': self.routing_table[dest_id].destSeqNum,
                    'valid': self.routing_table[dest_id].valid,
                    'precursorList': self.routing_table[dest_id].precursorList,
                    'lifeTime': self.routing_table[dest_id].lifeTime
                    }
                    self.send_sdn_route_update(self.sdn_node_num, route_info_data)'''



    