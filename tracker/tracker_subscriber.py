#!/usr/bin/env python
# -*- coding: utf-8 -*-

import zmq, json, socket,signal
import threading
from facenet_protocol import FacenetProtocol

"This Module is mZMQ Lib"
class TrackerSubscriber(threading.Thread):
    def __init__(self,addr):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(addr)
        self.socket.setsockopt(zmq.SUBSCRIBE, '')
        self.thread_state = True
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.msg_list = []
    def stop(self):
        print("zmq recv thread stop!")
        self.thread_state = False
        signal.alarm(1) #break select method

    def run(self):
        facemsg = FacenetProtocol()
        while self.thread_state:
            socks = dict(self.poller.poll())
            if socks.get(self.socket) == zmq.POLLIN:
                msg_cli = self.socket.recv()
                recv = facemsg.parserFaceData(msg_cli)
                print(recv)
            #self.msg_list.append(recv)



def sigint_handler(signum,frame):
    print("main-thread exit")
    global subs
    subs.stop()
    sys.exit()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    subs = TrackerSubscriber("tcp://127.0.0.1:812354")
    subs.start()
    subs.join()

