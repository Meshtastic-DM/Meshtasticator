#!/usr/bin/env python3
import math
import random

import simpy

from lib.common import calc_dist, find_random_position
from lib.mac import set_transmit_delay, get_retransmission_msec
from lib.phy import check_collision, is_channel_active, airtime
from lib.packet import NODENUM_BROADCAST, MeshPacket, MeshMessage


class MeshNode:
    def __init__(self, conf, nodes, env, bc_pipe, nodeid, period, messages, packetsAtN, packets, delays, nodeConfig, messageSeq, verboseprint):
        self.conf = conf
        self.nodeid = nodeid
        self.verboseprint = verboseprint
        self.moveRng = random.Random(nodeid)
        self.nodeRng = random.Random(nodeid)
        self.rebroadcastRng = random.Random()
        if nodeConfig is not None:
            self.x = nodeConfig['x']
            self.y = nodeConfig['y']
            self.z = nodeConfig['z']
            self.isRouter = nodeConfig['isRouter']
            self.isRepeater = nodeConfig['isRepeater']
            self.isClientMute = nodeConfig['isClientMute']
            self.hopLimit = nodeConfig['hopLimit']
            self.antennaGain = nodeConfig['antennaGain']
        else:
            self.x, self.y = find_random_position(self.conf, nodes)
            self.z = self.conf.HM
            self.isRouter = self.conf.router
            self.isRepeater = False
            self.isClientMute = False
            self.hopLimit = self.conf.hopLimit
            self.antennaGain = self.conf.GL
        self.messageSeq = messageSeq
        self.env = env
        self.period = period
        self.bc_pipe = bc_pipe
        self.nodes = nodes
        self.messages = messages
        self.packetsAtN = packetsAtN
        self.nrPacketsSent = 0
        self.packets = packets
        self.delays = delays
        self.timesReceived = {}
        self.isReceiving = []
        self.isTransmitting = False
        self.usefulPackets = 0
        self.txAirUtilization = 0
        self.airUtilization = 0
        self.droppedByDelay = 0
        self.rebroadcastPackets = 0
        self.isMoving = False
        self.gpsEnabled = False
        
        # ZRP-specific data structures
        self.neighbors = {}  # {nodeId: last_heard_time}
        self.routing_table = {}  # {destId: {nextHop: nodeId, hopCount: int, timestamp: time}}
        self.zone_nodes = set()  # Nodes within zone radius
        self.pending_route_requests = {}  # {destId: {seq: int, timestamp: time}}
        self.route_request_cache = {}  # {(origId, destId, seq): timestamp} to prevent loops
        # Track last broadcast position/time
        self.lastBroadcastX = self.x
        self.lastBroadcastY = self.y
        self.lastBroadcastTime = 0
        # track total transmit time for the last 6 buckets (each is 10s in firmware logic)
        self.channelUtilization = [0] * self.conf.CHANNEL_UTILIZATION_PERIODS  # each entry is ms spent on air in that interval
        self.channelUtilizationIndex = 0  # which "bucket" is current
        self.prevTxAirUtilization = 0.0   # how much total tx air-time had been used at last sample

        env.process(self.track_channel_utilization(env))
        if not self.isRepeater:  # repeaters don't generate messages themselves
            env.process(self.generate_message())
        env.process(self.receive(self.bc_pipe.get_output_conn()))
        self.transmitter = simpy.Resource(env, 1)
        
        # Start ZRP-specific processes if ZRP is enabled
        if self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.ZRP:
            env.process(self.zrp_hello_process())
            env.process(self.zrp_maintenance_process())

        # start mobility if enabled
        if self.conf.MOVEMENT_ENABLED and self.moveRng.random() <= self.conf.APPROX_RATIO_NODES_MOVING:
            self.isMoving = True
            if self.moveRng.random() <= self.conf.APPROX_RATIO_OF_NODES_MOVING_W_GPS_ENABLED:
                self.gpsEnabled = True

            # Randomly assign a movement speed
            possibleSpeeds = [
                self.conf.WALKING_METERS_PER_MIN,  # e.g.,  96 m/min
                self.conf.BIKING_METERS_PER_MIN,   # e.g., 390 m/min
                self.conf.DRIVING_METERS_PER_MIN   # e.g., 1500 m/min
            ]
            self.movementStepSize = self.moveRng.choice(possibleSpeeds)

            env.process(self.move_node(env))

    def track_channel_utilization(self, env):
        """
        Periodically compute how many seconds of airtime this node consumed
        over the last 10-second block and store it in the ring buffer.
        """
        while True:
            # Wait 10 seconds of simulated time
            yield env.timeout(self.conf.TEN_SECONDS_INTERVAL)

            curTotalAirtime = self.txAirUtilization  # total so far, in *milliseconds*
            blockAirtimeMs = curTotalAirtime - self.prevTxAirUtilization

            self.channelUtilization[self.channelUtilizationIndex] = blockAirtimeMs

            self.prevTxAirUtilization = curTotalAirtime
            self.channelUtilizationIndex = (self.channelUtilizationIndex + 1) % self.conf.CHANNEL_UTILIZATION_PERIODS

    def channel_utilization_percent(self) -> float:
        """
        Returns how much of the last 60 seconds (6 x 10s) this node spent transmitting, as a percent.
        """
        sumMs = sum(self.channelUtilization)
        # 6 intervals, each 10 seconds = 60,000 ms total
        # fraction = sum_ms / 60000, then multiply by 100 for percent
        return (sumMs / (self.conf.CHANNEL_UTILIZATION_PERIODS * self.conf.TEN_SECONDS_INTERVAL)) * 100.0

    def move_node(self, env):
        while True:

            # Pick a random direction and distance
            angle = 2 * math.pi * self.moveRng.random()
            distance = self.movementStepSize * self.moveRng.random()

            # Compute new position
            dx = distance * math.cos(angle)
            dy = distance * math.sin(angle)

            leftBound = self.conf.OX - self.conf.XSIZE / 2
            rightBound = self.conf.OX + self.conf.XSIZE / 2
            bottomBound = self.conf.OY - self.conf.YSIZE / 2
            topBound = self.conf.OY + self.conf.YSIZE / 2

            # Then in moveNode:
            new_x = min(max(self.x + dx, leftBound), rightBound)
            new_y = min(max(self.y + dy, bottomBound), topBound)

            # Update node’s position
            self.x = new_x
            self.y = new_y

            if self.gpsEnabled:
                distanceTraveled = calc_dist(self.lastBroadcastX, self.x, self.lastBroadcastY, self.y)
                timeElapsed = env.now - self.lastBroadcastTime
                if distanceTraveled >= self.conf.SMART_POSITION_DISTANCE_THRESHOLD and timeElapsed >= self.conf.SMART_POSITION_DISTANCE_MIN_TIME:
                    currentUtil = self.channel_utilization_percent()
                    if currentUtil < 25.0:
                        self.send_packet(NODENUM_BROADCAST, "POSITION")
                        self.lastBroadcastX = self.x
                        self.lastBroadcastY = self.y
                        self.lastBroadcastTime = env.now
                    else:
                        self.verboseprint(f"At time {env.now} node {self.nodeid} SKIPS POSITION broadcast (util={currentUtil:.1f}% > 25%)")

            # Wait until next move
            nextMove = self.get_next_time(self.conf.ONE_MIN_INTERVAL)
            if nextMove >= 0:
                yield self.env.timeout(nextMove)
            else:
                break
    
    def zrp_hello_process(self):
        """Periodically send HELLO packets for neighbor discovery"""
        while True:
            yield self.env.timeout(self.conf.ZRP_HELLO_INTERVAL)
            self.send_hello_packet()
    
    def zrp_maintenance_process(self):
        """Periodic maintenance of neighbors and routing tables"""
        while True:
            yield self.env.timeout(self.conf.ZRP_NEIGHBOR_TIMEOUT // 2)
            self.cleanup_stale_neighbors()
            self.update_zone_nodes()
    
    def send_hello_packet(self):
        """Send HELLO packet to announce presence"""
        self.messageSeq["val"] += 1
        messageSeq = self.messageSeq["val"]
        hello_packet = MeshPacket(
            self.conf, self.nodes, self.nodeid, NODENUM_BROADCAST, self.nodeid,
            10, messageSeq, self.env.now, False, False, None, self.env.now, self.verboseprint
        )
        hello_packet.packet_type = "HELLO"
        self.packets.append(hello_packet)
        self.env.process(self.transmit(hello_packet))
    
    def cleanup_stale_neighbors(self):
        """Remove neighbors that haven't been heard from recently"""
        current_time = self.env.now
        stale_neighbors = [nid for nid, last_time in self.neighbors.items() 
                          if current_time - last_time > self.conf.ZRP_NEIGHBOR_TIMEOUT]
        for nid in stale_neighbors:
            del self.neighbors[nid]
            # Remove routes through stale neighbors
            routes_to_remove = [dest for dest, route in self.routing_table.items() 
                              if route['nextHop'] == nid]
            for dest in routes_to_remove:
                del self.routing_table[dest]
    
    def update_zone_nodes(self):
        """Update the set of nodes within zone radius"""
        self.zone_nodes.clear()
        self.zone_nodes.add(self.nodeid)  # Include self
        # Add direct neighbors
        for neighbor_id in self.neighbors.keys():
            self.zone_nodes.add(neighbor_id)
        # Add nodes reachable within zone radius using BFS
        visited = set([self.nodeid])
        queue = [(neighbor_id, 1) for neighbor_id in self.neighbors.keys()]
        while queue:
            node_id, hops = queue.pop(0)
            if hops <= self.conf.ZRP_ZONE_RADIUS and node_id not in visited:
                visited.add(node_id)
                self.zone_nodes.add(node_id)
                # Add this node's neighbors for next level
                node = next((n for n in self.nodes if n.nodeid == node_id), None)
                if node:
                    for next_neighbor in node.neighbors.keys():
                        if next_neighbor not in visited and hops + 1 <= self.conf.ZRP_ZONE_RADIUS:
                            queue.append((next_neighbor, hops + 1))
    
    def zrp_route_discovery(self, dest_id):
        """Initiate route discovery for destination outside zone"""
        if dest_id in self.pending_route_requests:
            return  # Already discovering route
        
        self.messageSeq["val"] += 1
        seq = self.messageSeq["val"]
        self.pending_route_requests[dest_id] = {'seq': seq, 'timestamp': self.env.now}
        
        # Send RREQ to zone border nodes
        for border_node in self.get_zone_border_nodes():
            rreq_packet = MeshPacket(
                self.conf, self.nodes, self.nodeid, border_node, self.nodeid,
                20, seq, self.env.now, False, False, None, self.env.now, self.verboseprint
            )
            rreq_packet.packet_type = "RREQ"
            rreq_packet.dest_target = dest_id
            rreq_packet.hop_count = 0
            self.packets.append(rreq_packet)
            self.env.process(self.transmit(rreq_packet))
    
    def get_zone_border_nodes(self):
        """Get nodes at the border of the zone"""
        border_nodes = set()
        for node_id in self.zone_nodes:
            node = next((n for n in self.nodes if n.nodeid == node_id), None)
            if node:
                for neighbor_id in node.neighbors.keys():
                    if neighbor_id not in self.zone_nodes:
                        border_nodes.add(node_id)
                        break
        return border_nodes
    
    def zrp_intrazone_route(self, dest_id):
        """Find route within zone using proactive routing"""
        if dest_id in self.routing_table:
            route = self.routing_table[dest_id]
            if self.env.now - route['timestamp'] < self.conf.ZRP_ROUTE_TIMEOUT:
                return route['nextHop']
        return None

    def send_packet(self, destId, type=""):
        # increment the shared counter
        self.messageSeq["val"] += 1
        messageSeq = self.messageSeq["val"]
        self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))
        p = MeshPacket(self.conf, self.nodes, self.nodeid, destId, self.nodeid, self.conf.PACKETLENGTH, messageSeq, self.env.now, True, False, None, self.env.now, self.verboseprint)
        self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'generated', type, 'message', p.seq, 'to', destId)
        self.packets.append(p)
        self.env.process(self.transmit(p))
        return p

    def get_next_time(self, period):
        nextGen = self.nodeRng.expovariate(1.0 / float(period))
        # do not generate message near the end of the simulation (otherwise flooding cannot finish in time)
        if self.env.now+nextGen + self.hopLimit * airtime(self.conf, self.conf.SFMODEM[self.conf.MODEM], self.conf.CRMODEM[self.conf.MODEM], self.conf.PACKETLENGTH, self.conf.BWMODEM[self.conf.MODEM]) < self.conf.SIMTIME:
            return nextGen
        return -1
    

    def was_seen_recently(self, packet, ownTransmit=False):
        if packet.seq not in self.timesReceived:
            # First time we know about this packet
            self.timesReceived[packet.seq] = 0 if ownTransmit else 1
            if not ownTransmit:
                self.usefulPackets += 1
        else:
            self.timesReceived[packet.seq] += 0 if ownTransmit else 1


    def perhaps_cancel_dupe(self, packet):
        # Cancel if we've already seen this sequence number
        if packet.seq in self.timesReceived:
            return self.timesReceived[packet.seq] > 2 if self.isRouter or self.isRepeater else self.timesReceived[packet.seq] > 1
        return False


    def generate_message(self):
        while True:
            # Returns -1 if we don't make it before the sim ends
            nextGen = self.get_next_time(self.period)
            # do not generate a message near the end of the simulation (otherwise flooding cannot finish in time)
            if nextGen >= 0:
                yield self.env.timeout(nextGen)

                if self.conf.DMs:
                    destId = self.nodeRng.choice([i for i in range(0, len(self.nodes)) if i is not self.nodeid])
                else:
                    destId = NODENUM_BROADCAST

                p = self.send_packet(destId)

                while p.wantAck:  # ReliableRouter: retransmit message if no ACK received after timeout
                    retransmissionMsec = get_retransmission_msec(self, p)
                    yield self.env.timeout(retransmissionMsec)

                    ackReceived = False  # check whether you received an ACK on the transmitted message
                    minRetransmissions = self.conf.maxRetransmission
                    for packetSent in self.packets:
                        if packetSent.origTxNodeId == self.nodeid and packetSent.seq == p.seq:
                            if packetSent.retransmissions < minRetransmissions:
                                minRetransmissions = packetSent.retransmissions
                            if packetSent.ackReceived:
                                ackReceived = True
                    if ackReceived:
                        self.verboseprint('Node', self.nodeid, 'received ACK on generated message with seq. nr.', p.seq)
                        break
                    else:
                        if minRetransmissions > 0:  # generate new packet with same sequence number
                            pNew = MeshPacket(self.conf, self.nodes, self.nodeid, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now, self.verboseprint)
                            pNew.retransmissions = minRetransmissions - 1
                            self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'wants to retransmit its generated packet to', destId, 'with seq.nr.', p.seq, 'minRetransmissions', minRetransmissions)
                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                        else:
                            self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'reliable send of', p.seq, 'failed.')
                            break
            else:  # do not send this message anymore, since it is close to the end of the simulation
                break

    def transmit(self, packet):
        with self.transmitter.request() as request:
            yield request

            # listen-before-talk from src/mesh/RadioLibInterface.cpp
            txTime = set_transmit_delay(self, packet)
            self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'picked wait time', txTime)
            yield self.env.timeout(txTime)

            # wait when currently receiving or transmitting, or channel is active
            while any(self.isReceiving) or self.isTransmitting or is_channel_active(self, self.env):
                self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'is busy Tx-ing', self.isTransmitting, 'or Rx-ing', any(self.isReceiving), 'else channel busy!')
                txTime = set_transmit_delay(self, packet)
                yield self.env.timeout(txTime)
            self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'ends waiting')

            # check if you received an ACK for this message in the meantime
            self.was_seen_recently(packet, ownTransmit=True)
            if not self.perhaps_cancel_dupe(packet):  # if you did not receive an ACK for this message in the meantime
                self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'started low level send', packet.seq, 'hopLimit', packet.hopLimit, 'original Tx', packet.origTxNodeId)
                self.nrPacketsSent += 1
                for rx_node in self.nodes:
                    if packet.sensedByN[rx_node.nodeid]:
                        if check_collision(self.conf, self.env, packet, rx_node.nodeid, self.packetsAtN) == 0:
                            self.packetsAtN[rx_node.nodeid].append(packet)
                packet.startTime = self.env.now
                packet.endTime = self.env.now + packet.timeOnAir
                self.txAirUtilization += packet.timeOnAir
                self.airUtilization += packet.timeOnAir
                self.bc_pipe.put(packet)
                self.isTransmitting = True
                yield self.env.timeout(packet.timeOnAir)
                self.isTransmitting = False
            else:  # received ACK: abort transmit, remove from packets generated
                self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'in the meantime received ACK, abort packet with seq. nr', packet.seq)
                self.packets.remove(packet)

    def receive(self, in_pipe):
        while True:
            p = yield in_pipe.get()
            if p.sensedByN[self.nodeid] and not p.collidedAtN[self.nodeid] and p.onAirToN[self.nodeid]:  # start of reception
                if not self.isTransmitting:
                    self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'started receiving packet', p.seq, 'from', p.txNodeId)
                    p.onAirToN[self.nodeid] = False
                    self.isReceiving.append(True)
                else:  # if you were currently transmitting, you could not have sensed it
                    self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'was transmitting, so could not receive packet', p.seq)
                    p.sensedByN[self.nodeid] = False
                    p.onAirToN[self.nodeid] = False
            elif p.sensedByN[self.nodeid]:  # end of reception
                try:
                    self.isReceiving[self.isReceiving.index(True)] = False
                except Exception:
                    pass
                self.airUtilization += p.timeOnAir
                if p.collidedAtN[self.nodeid]:
                    self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'could not decode packet.')
                    continue
                p.receivedAtN[self.nodeid] = True
                self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'received packet', p.seq, 'with delay', round(self.env.now - p.genTime, 2))
                self.delays.append(self.env.now - p.genTime)

                # Update history of received packets
                self.was_seen_recently(p)

                # check if implicit ACK for own generated message
                if p.origTxNodeId == self.nodeid:
                    if p.isAck:
                        self.verboseprint('Node', self.nodeid, 'received real ACK on generated message.')
                    else:
                        self.verboseprint('Node', self.nodeid, 'received implicit ACK on message sent.')
                    p.ackReceived = True
                    continue

                ackReceived = False
                realAckReceived = False
                for sentPacket in self.packets:
                    # check if ACK for message you currently have in queue
                    if sentPacket.txNodeId == self.nodeid and sentPacket.seq == p.seq:
                        self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'received implicit ACK for message in queue.')
                        ackReceived = True
                        sentPacket.ackReceived = True
                    # check if real ACK for message sent
                    if sentPacket.origTxNodeId == self.nodeid and p.isAck and sentPacket.seq == p.requestId:
                        self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'received real ACK.')
                        realAckReceived = True
                        sentPacket.ackReceived = True

                # send real ACK if you are the destination and you did not yet send the ACK
                if p.wantAck and p.destId == self.nodeid and not any(pA.requestId == p.seq for pA in self.packets):
                    self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'sends a flooding ACK.')
                    self.messageSeq["val"] += 1
                    messageSeq = self.messageSeq["val"]
                    self.messages.append(MeshMessage(self.nodeid, p.origTxNodeId, self.env.now, messageSeq))
                    pAck = MeshPacket(self.conf, self.nodes, self.nodeid, p.origTxNodeId, self.nodeid, self.conf.ACKLENGTH, messageSeq, self.env.now, False, True, p.seq, self.env.now, self.verboseprint)
                    self.packets.append(pAck)
                    self.env.process(self.transmit(pAck))
                # Handle special ZRP packets
                if hasattr(p, 'packet_type'):
                    if p.packet_type == "HELLO":
                        self.neighbors[p.txNodeId] = self.env.now
                        self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'updated neighbor', p.txNodeId)
                        continue
                    elif p.packet_type == "RREQ" and hasattr(p, 'dest_target'):
                        self.handle_route_request(p)
                        continue
                    elif p.packet_type == "RREP" and hasattr(p, 'dest_target'):
                        self.handle_route_reply(p)
                        continue
                
                # Rebroadcasting Logic for received message. This is a broadcast or a DM not meant for us.
                elif not p.destId == self.nodeid and not ackReceived and not realAckReceived and p.hopLimit > 0:
                    if self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.ZRP:
                        # ZRP routing: use different strategies for broadcast vs DM
                        if p.destId == NODENUM_BROADCAST:
                            # Use managed flooding for broadcast messages
                            if not self.isClientMute:
                                self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'ZRP: flooding broadcast packet', p.seq)
                                pNew = MeshPacket(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now, self.verboseprint)
                                pNew.hopLimit = p.hopLimit - 1
                                self.packets.append(pNew)
                                self.env.process(self.transmit(pNew))
                        else:
                            # Use ZRP for DM messages
                            next_hop = self.zrp_get_next_hop(p.destId)
                            if next_hop is not None:
                                self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'ZRP: forwarding DM packet', p.seq, 'to next hop', next_hop)
                                pNew = MeshPacket(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now, self.verboseprint)
                                pNew.hopLimit = p.hopLimit - 1
                                # Only forward if next hop can receive the packet
                                next_hop_node = next((n for n in self.nodes if n.nodeid == next_hop), None)
                                if next_hop_node and pNew.sensedByN[next_hop]:
                                    self.packets.append(pNew)
                                    self.env.process(self.transmit(pNew))
                            else:
                                # No route known, initiate route discovery if within zone or drop
                                if p.destId in self.zone_nodes:
                                    # Should have route within zone - something is wrong
                                    self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'ZRP: no route to', p.destId, 'within zone, dropping')
                                else:
                                    # Start route discovery for nodes outside zone
                                    self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'ZRP: initiating route discovery for', p.destId)
                                    self.zrp_route_discovery(p.destId)
                    # FloodingRouter: rebroadcast received packet
                    elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.MANAGED_FLOOD:
                        if not self.isClientMute:
                            self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'rebroadcasts received packet', p.seq)
                            pNew = MeshPacket(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now, self.verboseprint)
                            pNew.hopLimit = p.hopLimit - 1
                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                else:
                    self.droppedByDelay += 1
    
    def zrp_get_next_hop(self, dest_id):
        """Get next hop for destination using ZRP"""
        # First check if destination is within zone (proactive routing)
        if dest_id in self.zone_nodes:
            next_hop = self.zrp_intrazone_route(dest_id)
            if next_hop is not None:
                return next_hop
        
        # Check if we have a cached route (from previous route discovery)
        if dest_id in self.routing_table:
            route = self.routing_table[dest_id]
            if self.env.now - route['timestamp'] < self.conf.ZRP_ROUTE_TIMEOUT:
                return route['nextHop']
        
        return None
    
    def handle_route_request(self, rreq_packet):
        """Handle incoming route request"""
        dest_id = rreq_packet.dest_target
        orig_id = rreq_packet.origTxNodeId
        seq = rreq_packet.seq
        
        # Check for duplicate RREQ
        rreq_key = (orig_id, dest_id, seq)
        if rreq_key in self.route_request_cache:
            return
        
        self.route_request_cache[rreq_key] = self.env.now
        
        # If we are the destination, send RREP
        if dest_id == self.nodeid:
            self.send_route_reply(orig_id, seq, 0)
            return
        
        # If destination is in our zone, send RREP with route
        if dest_id in self.zone_nodes:
            next_hop = self.zrp_intrazone_route(dest_id)
            if next_hop is not None:
                hop_count = self.routing_table.get(dest_id, {}).get('hopCount', 1)
                self.send_route_reply(orig_id, seq, hop_count + rreq_packet.hop_count)
                return
        
        # Forward RREQ to border nodes if we are a border node
        if self.nodeid in self.get_zone_border_nodes():
            for border_node in self.get_zone_border_nodes():
                if border_node != rreq_packet.txNodeId:  # Don't send back to sender
                    new_rreq = MeshPacket(
                        self.conf, self.nodes, orig_id, border_node, self.nodeid,
                        20, seq, rreq_packet.genTime, False, False, None, self.env.now, self.verboseprint
                    )
                    new_rreq.packet_type = "RREQ"
                    new_rreq.dest_target = dest_id
                    new_rreq.hop_count = rreq_packet.hop_count + 1
                    self.packets.append(new_rreq)
                    self.env.process(self.transmit(new_rreq))
    
    def send_route_reply(self, orig_id, seq, hop_count):
        """Send route reply back to originator"""
        rrep_packet = MeshPacket(
            self.conf, self.nodes, self.nodeid, orig_id, self.nodeid,
            15, seq, self.env.now, False, False, None, self.env.now, self.verboseprint
        )
        rrep_packet.packet_type = "RREP"
        rrep_packet.dest_target = orig_id
        rrep_packet.hop_count = hop_count
        self.packets.append(rrep_packet)
        self.env.process(self.transmit(rrep_packet))
    
    def handle_route_reply(self, rrep_packet):
        """Handle incoming route reply"""
        dest_id = rrep_packet.origTxNodeId  # Original destination
        next_hop = rrep_packet.txNodeId     # Next hop to reach destination
        hop_count = rrep_packet.hop_count
        
        # Update routing table
        self.routing_table[dest_id] = {
            'nextHop': next_hop,
            'hopCount': hop_count,
            'timestamp': self.env.now
        }
        
        self.verboseprint(round(self.env.now, 3), 'Node', self.nodeid, 'learned route to', dest_id, 'via', next_hop)
        
        # Remove from pending requests
        if dest_id in self.pending_route_requests:
            del self.pending_route_requests[dest_id]
