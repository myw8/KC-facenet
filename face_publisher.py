#!/usr/bin/python
#-*-coding:utf-8-*-
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import zmq, json, signal,time
import threading
class FacePublisher(threading.Thread):
    def __init__(self,addr):
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.setsockopt(zmq.SNDBUF, 20000 * 1024)
        self.addr = addr
        #self.publisher.bind(addr)
        self.thread_state = True
        self.list_face = []
        self.mutex = threading.Lock()

    def insertListFace(self,dict_face):
        self.mutex.acquire()
        self.list_face.append(dict_face)
        self.mutex.release()

    def send(self, msg):
        jsonmsg = json.dumps(msg)
        #print(jsonmsg)
        self.publisher.send_json(jsonmsg)

    def stop(self):
        print("FacePublisher thread stop!")
        self.thread_state = False
        self.publisher.close()
        signal.alarm(1) #break select method

    ######################################################
    def run(self):
        self.publisher.bind(self.addr)
        while self.thread_state:
            while len(self.list_face)>0:
                self.send(self.list_face.pop())
            time.sleep(0.01)
    ######################################################

'''
def sigint_handler(signum,frame):
    print("main-thread exit")
    global publish
    publish.stop()
    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    publish = FacePublisher("tcp://127.0.0.1:12347")
    publish.start()
    publish.join()
'''


