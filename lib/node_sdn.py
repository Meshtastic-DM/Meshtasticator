from lib.node_aodv import MeshNode_AODV
import json
import os
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
       

    def update_routing_table(self, destId, nextHop, hopCount, destSeqNum, valid=True, precursorList=None, lifeTime=300000):
        # Normalize precursorList to a serializable list
        pl = precursorList if precursorList is not None else []
        if self.sdn_node_num is not None and self.simRole != 'sdn_node' and hopCount <=2:
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
        # Pass the normalized precursor list to the base implementation
        super().update_routing_table(destId, nextHop, hopCount, destSeqNum, valid, pl, lifeTime)

    def send_sdn_route_update(self, controller_node_num, route_info_data):
        if controller_node_num is not None and self.simRole == "DM":
            self.send_packet(controller_node_num, data=route_info_data, is_sdn_update=True)
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'sent SDN route update to controller node', controller_node_num)

    def handle_sdn_update(self, packet):
        if packet.seq in self.processed_sdn_update_packets:
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
                    next_hop = self.routing_table.get(packet.destId).nextHop if packet.destId else None
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
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'forwarded SDN update packet', pNew.seq, 'towards', packet.destId, 'with next hop', next_hop)
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
        updated = False
        if self.sdn_node_num is None:
            self.sdn_node_num = packet.origTxNodeId
            self.sdn_node_hop_count = packet.hop_count
            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'set SDN controller to node', self.sdn_node_num, 'at hop count', self.sdn_node_hop_count)
            updated = True
        else:
            if packet.hop_count < self.sdn_node_hop_count:
                self.sdn_node_num = packet.origTxNodeId
                self.sdn_node_hop_count = packet.hop_count
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'updated SDN controller to node', self.sdn_node_num, 'at hop count', self.sdn_node_hop_count)
                updated = True
        if updated:
            self.update_routing_table(self.sdn_node_num, packet.txNodeId, packet.hop_count+1, packet.seq, valid=True)
            for dest_id in self.routing_table.keys():
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
                    self.send_sdn_route_update(self.sdn_node_num, route_info_data)



    