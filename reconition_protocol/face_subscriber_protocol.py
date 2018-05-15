#!/usr/bin/env python
# -*- coding: utf-8 -*-

import zmq, json, socket,signal
import threading
import time
import sys
import pickle
class FaceSubscriber(threading.Thread):
    def __init__(self,addr,pushaddr,yoloq):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(addr)
        self.socket.setsockopt(zmq.SUBSCRIBE, '')
        self.thread_state = True
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        self.yoloq = yoloq
        self.mutex = threading.Lock()
        self.sender = self.context.socket(zmq.PUB)
        self.sender.bind(pushaddr)
        self.sender.send(str(2))
    def stop(self):
        print("FaceSubscriber thread stop!")
        self.thread_state = False
        self.socket.close()
        #signal.alarm(1) #break select method

    def run(self):
        i = 0
        while self.thread_state:
            socks = dict(self.poller.poll())
            if socks.get(self.socket) == zmq.POLLIN:
                msg_cli = self.socket.recv()

                if(i ==0):
                    pickle.dump(msg_cli, open('yolodata', 'wb'))
                    i =2
                #recv = json.loads(msg_cli)
                self.yoloq.put(msg_cli)
                self.sender.send(str(2))
                print('yoloq len = %d ###########################'%(self.yoloq.qsize()))
                time.sleep(0.1)
            #print(recv)





