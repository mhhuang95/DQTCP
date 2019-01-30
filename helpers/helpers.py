#adjusted from https://github.com/StanfordSNR/indigo/blob/master/helpers/helpers.py

import os
import time
import select
import socket
import numpy as np
import operator
import errno


READ_FLAGS = select.POLLIN | select.POLLPRI
WRITE_FLAGS = select.POLLOUT
ERR_FLAGS = select.POLLERR | select.POLLHUP | select.POLLNVAL
READ_ERR_FLAGS = READ_FLAGS | ERR_FLAGS
ALL_FLAGS = READ_FLAGS | WRITE_FLAGS | ERR_FLAGS


math_ops = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}

def apply_op(op, op1, op2):
    return math_ops[op](op1, op2)

def curr_ts_ms():
    if not hasattr(curr_ts_ms, 'epoch'):
        curr_ts_ms.epoch = time.time()

    return int((time.time() - curr_ts_ms.epoch) * 1000)


def get_open_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    s.bind(('', 0))
    #getsockname():Return the socket's own address. This is useful to find out the port number of an IPv4/v6 socket, for instance
    port = s.getsockname()[1]
    s.close()
    return port