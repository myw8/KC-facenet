#!/usr/bin/python
#-*-coding:utf-8-*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import zmq, json, signal,time
import random
import threading
from yolomsg import YoloMsg
import sys
class Publisher(threading.Thread):
    def __init__(self,addr):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(addr)
        self.thread_state = True
        self.yolomsg = YoloMsg()
    def send(self, msg):
        jsonmsg = json.dumps(msg)
        print(jsonmsg)
        self.publisher.send_json(jsonmsg)

    def stop(self):
        print("zmq recv thread stop!")
        self.thread_state = False
        signal.alarm(1) #break select method

    ######################################################
    def run(self):
        while self.thread_state:
            if self.yolomsg.hasYoloMsg()>0:
                self.send(self.yolomsg.PopMsg())
            time.sleep(1)
    ######################################################


def sigint_handler(signum,frame):
    print("main-thread exit")
    global publish
    publish.stop()
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    publish = Publisher("tcp://127.0.0.1:12347")
    publish.start()
    publish.join()

