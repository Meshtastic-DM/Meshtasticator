from lib.node_aodv import MeshNode_AODV
import json
import os

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
        self.sdn_controller_node_num = 0

    def update_routing_table(self, destId, nextHop, hopCount, destSeqNum, valid=True, precursorList=None, lifeTime=300000):
        # Normalize precursorList to a serializable list
        pl = precursorList if precursorList is not None else []
        if self.sdn_controller_node_num is not None and self.simRole != 'sdn_node' and hopCount ==1:
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
            self.send_sdn_route_update(self.sdn_controller_node_num, route_info_data)
        # Pass the normalized precursor list to the base implementation
        super().update_routing_table(destId, nextHop, hopCount, destSeqNum, valid, pl, lifeTime)

    def send_sdn_route_update(self, controller_node_num, route_info_data):
        self.send_packet(controller_node_num, data=route_info_data, is_sdn_update=True)
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'sent SDN route update to controller node', controller_node_num)

    def handle_sdn_update(self, packet):
        if self.simRole == 'sdn_node':
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