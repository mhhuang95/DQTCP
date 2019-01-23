#adjusted from https://github.com/StanfordSNR/indigo/blob/master/env/sender.py
#Minhui Huang

import time
import sys
import json
import socket
import select
from os import path
import numpy as np
import datagram_pb2
import project_root
from helpers.helpers import (
    curr_ts_ms, apply_op,
    READ_FLAGS, ERR_FLAGS, READ_ERR_FLAGS, WRITE_FLAGS, ALL_FLAGS)


def format_actions(action_list):
    """ Returns the action list, initially a list with elements "[op][val]"
    like /2.0, -3.0, +1.0, formatted as a dictionary.
    The dictionary keys are the unique indices (to retrieve the action) and
    the values are lists ['op', val], such as ['+', '2.0'].
    """
    return {idx: [action[0], float(action[1:])]
                  for idx, action in enumerate(action_list)}


class Sender(object):
    #initialization
    max_steps = 1000
    state_dim = 4
    action_mapping = format_actions(["/2.0", "-10.0", "+0.0", "+10.0", "*2.0"])
    action_cnt = len(action_mapping)

    def __init__(self, port=0, train=False, debug=False):
        self.train = train
        self.debug = debug

        #UDP socket and poller
        self.peer_addr = None

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # The setsockopt() function shall set the option specified by the option_name argument,
        # SO_REUSEADDR: Specifies that the rules used in validating addresses supplied to bind() should allow reuse of local addresses,
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        #Bind the socket to address.
        self.sock.bind(('0.0.0.0', port))

        #getsockname():Return the socket’s own address.
        sys.stderr.write('[sender] Listening on port %s\n' %self.sock.getsockname()[1])

        # Returns a polling object, which supports registering and unregistering file descriptors, and then polling them for I/O events
        self.poller = select.poll()
        #Register a file descriptor with the polling object.
        self.poller.register(self.sock, ALL_FLAGS)

        self.dummy_payload = 'x' * 1400

        if self.debug:
            self.sampling_file = open(path.join(project_root.DIR, 'env', 'sampling_time'), 'w', 0)


        #congestion control related
        self.seq_num = 0
        self.next_ack = 0
        self.cwnd = 10.0
        self.step_len_ms = 10

        #state variable for RLCC
        self.delivered_time = 0
        self.delivered = 0
        self.sent_bytes = 0

        self.min_rtt = float('inf')

        self.delay_ewma = None
        self.send_rate_ewma = None
        self.delivery_rate_ewma = None

        self.step_start_ms = None
        self.running = True

        if self.train:
            self.step_cnt = 0

            self.ts_first = None
            self.rtt_buf = []

    def clean_up(self):
        if self.debug and self.sampling_file:
            self.sampling_file.close()
        self.sock.close()

    def handshake(self):
        #handshake with peer receiver. Must be called before run()

        while True:
            #Receive data from the socket. The return value is a pair (string, address)
            # where string is a string representing the data received and address is the address of the socket sending the data.
            msg, addr = self.sock.recvfrom(1600)

            if msg == 'Hello form receiver' and self.peer_addr == None:
                self.peer_addr = addr
                self.sock.sendto('Hello from sender', self.peer_addr)
                sys.stderr.write('[sender] Handshake success! '
                                 'Receiver\'s address is %s:%s\n' % addr)
                break
        #setblocking(0):Set blocking or non-blocking mode of the socket: if flag is 0, the socket is set to non-blocking, else to blocking mode
        self.sock.setblocking(0) # non-blocking UDP socket

    def set_sample_action(self, sample_action):
        """Set the policy. Must be called before run()."""

        self.sample_action = sample_action

    