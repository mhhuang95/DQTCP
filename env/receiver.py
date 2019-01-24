#adjusted from https://github.com/StanfordSNR/indigo/blob/master/env/receiver.py


import sys
import json
import socket
import select
import datagram_pb2
import project_root
from helpers.helpers import READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, ALL_FLAGS

class Receiver(object):
    def __init__(self,ip,port):
        self.peer_addr = (ip, port)

        #UDP socket and poller
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)

        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

    def cleanup(self):
        self.sock.close()

    def construct_ack_from_data(self, serialized_data):
        """Construct a serialized ACK that acks a serialized datagram."""

        data = datagram_pb2.Data()
        data.ParseFromString(serialized_data)

        ack = datagram_pb2.Ack()
        ack.seq_num = data.seq_num
        ack.send_ts = data.send_ts
        ack.sent_bytes = data.sent_bytes
        ack.delivered_time = data.delivered_time
        ack.delivered = data.delivered
        ack.ack_bytes = len(serialized_data)

        return ack.SerializeToString()

    def handshake(self):










