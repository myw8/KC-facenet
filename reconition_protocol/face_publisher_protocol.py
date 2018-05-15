#!/usr/bin/python
#-*-coding:utf-8-*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import zmq, json, signal,time
import threading
class FacePublisher(threading.Thread):
    def __init__(self,addr,faceq):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.addr = addr
        self.thread_state = True
        self.faceq = faceq
        self.sender = self.context.socket(zmq.PUSH)


    def stop(self):
        print("FacePublisher thread stop!")
        self.thread_state = False
        self.publisher.close()


    ######################################################
    def run(self):
        self.publisher.bind(self.addr)
        while self.thread_state:
            self.publisher.send(self.faceq.get())






