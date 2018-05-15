# -*- coding: utf8 -*-
# ! /usr/bin/python
from FSRGAN.FSRGAN import LHFSRGAN
from SRGAN.SRGAN import LHSRGAN
import tensorlayer as tl
from multiprocessing import Process,Queue,Lock
import sys
from src import face
import threading
import time

class fsrgan_process(Process):
    def __init__(self,fsrq,recoq,lock):
        Process.__init__(self)
        self.fsrgan = LHFSRGAN()
        self.fsrq = fsrq
        self.recoq = recoq
        self.lock = lock
        self.state = True
    def run(self):
        while(self.state):
            if self.fsrq.qsize()>0:
                img = self.fsrq.get()
                out_img, _ = self.fsrgan.LR_to_HR(img)
                self.lock.acquire()
                self.recoq.put([out_img,1])
                self.lock.release()
                self.fsrq.task_done()
    def stop(self):
        self.state = False



class face_process(Process):
    def __init__(self,recoq,faceq):
        Process.__init__(self)
        self.face_recognition = face.Recognition()
        self.state = True
        self.recoq = recoq
        self.faceq = faceq

    def run(self):
        while(1):
         if self.recoq.qsize()>0:
             [img, flag] = self.recoq.get()
             faces = self.face_recognition.identify(img)
             self.faceq.put([faces[0].confidence, faces[0].name, flag])

    def stop(self):
        self.state = False

class saveTask(threading.Thread):
    def __init__(self,faceq):
        threading.Thread.__init__ (self)
        self.thread_state = True
        self.faceq = faceq

    def run(self):
        while self.thread_state:
            if self.faceq.qsize() > 0:
                [confidence, name, flag] = self.faceq.get()
                print(confidence, name, flag)
                time.sleep(0.1)

    def stop(self):
        self.thread_state = False

def sigint_handler(signum,frame):
    print("main-thread exit")
    global fsrgan
    global face_reco
    global savetask
    fsrgan.stop()
    face_reco.stop()
    savetask.stop()
    sys.exit()

if __name__ == '__main__':
    ## create folders to save result images
    fsrganqueue = Queue()
    recognitionqueue = Queue()
    facequeue = Queue()
    lock = Lock()
    face_reco = face_process(recognitionqueue, facequeue)
    fsrgan = fsrgan_process(fsrganqueue,recognitionqueue,lock)

    savetask = saveTask(facequeue)
    fsrgan.start()
    face_reco.start()
    savetask.start()
    out_img = tl.vis.read_image('samples/face_1_0002.jpg')
    if(out_img.shape() < (40,40)):
        fsrganqueue.put(out_img)
    else:
        recognitionqueue.put([out_img,0])







