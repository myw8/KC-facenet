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
        self.publisher.setsockopt(zmq.SNDBUF, 20000 * 1024)
        self.addr = addr
        #self.publisher.bind(addr)
        self.thread_state = True
        self.faceq = faceq
        self.sender = self.context.socket(zmq.PUSH)
        #self.sender.connect(pushaddr)

    def send(self, msg):
        #self.sender.send(2)
        jsonmsg = json.dumps(msg)
        #print(jsonmsg)
        self.publisher.send_json(jsonmsg)


    def stop(self):
        print("FacePublisher thread stop!")
        self.thread_state = False
        self.publisher.close()
        #signal.alarm(1) #break select method

    ######################################################
    def run(self):
        self.publisher.bind(self.addr)
        while self.thread_state:
            self.send(self.faceq.get())
            #self.faceq.task_done()





