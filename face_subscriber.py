#!/usr/bin/env python
# -*- coding: utf-8 -*-

import zmq, json, socket,signal
import threading
import time
import sys
"This Module is mZMQ Lib"
class FaceSubscriber(threading.Thread):
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
        self.mutex = threading.Lock()
    def stop(self):
        print("FaceSubscriber thread stop!")
        self.thread_state = False
        self.socket.close()
        signal.alarm(1) #break select method

    def run(self):
        while self.thread_state:
            #socks = dict(self.poller.poll())
            #if socks.get(self.socket) == zmq.POLLIN:
            msg_cli = self.socket.recv_json()
            recv = json.loads(msg_cli)
            self.msg_list.append(recv)
            print('msg_list len = %d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'%(len(self.msg_list)))
            time.sleep(0.1)
            #print(recv)

    def readMsg(self):
        msg = None
        if(len(self.msg_list)>0):
            self.mutex.acquire()
            msg = self.msg_list.pop()
            self.mutex.release()
        return  msg

    def hasMsg(self):
        return len(self.msg_list)


def sigint_handler(signum,frame):
    print("main-thread exit")
    global subs
    subs.stop()
    sys.exit()

'''

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    subs = FaceSubscriber("tcp://127.0.0.1:12347")
    subs.start()
    subs.join()
'''


