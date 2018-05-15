# -*- coding: utf8 -*-
# ! /usr/bin/python
from FSRGAN.FSRGAN import LHFSRGAN
from SRGAN.SRGAN import LHSRGAN
import tensorlayer as tl
from multiprocessing import Process,Queue,Lock
import sys
from src import face

from face_subscriber import FaceSubscriber
from face_publisher import FacePublisher
import sys,signal
import numpy as np
import cv2
import time
import threading
import time
import sys
import os


class FaceRecognition(Process):
#class FaceRecognition(threading.Thread):
    def __init__(self,q,yoloq,yololock,faceq,facelock):
        Process.__init__(self)
        #threading.Thread.__init__(self)
        self.thread_state = True
        self.face_recognition = None
        self.srgan = None
        #self.srgan = LHFSRGAN()

        self.q = q
        self.yoloq = yoloq
        self.yololock = yololock
        self.faceq = faceq
        self.facelock = facelock


    def stop(self):
        self.thread_state = False
        print("FaceRecognition %d Process stop!"%(os.getpid()))

        #signal.alarm(1) #break select method

    def yoloMsgProc(self,msg):
        thed = 0.6
        debug = 0
        for id, value in msg.items():
            print(id)
            dict_face = {}
            dict_face[id] = []
            item_tm = time.time()
            for i in range(len(value)):
                ##########################################
                D_ID = 0
                D_HEAD_AXIS = 1
                D_W = 2
                D_H = 3
                D_HEAD = 2
                person = value[i]
                # print(person)
                pid = str(person[D_ID])
                w = int(person[D_HEAD_AXIS][D_W])
                h = int(person[D_HEAD_AXIS][D_H])

                #print(pid, "@@@@@@@@@@w:%d h:%d" % (w, h))
                data = person[D_HEAD]
                arr = np.array(data)
                arr = arr.reshape(-1)
                try:
                    img = arr.reshape(h, w, 3)
                except:
                    continue
                img = np.uint8(img)
                faceimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
                if debug:
                    cv2.imshow("Face: " + str(i), img)
                # face rgb
                #########################################
                flag = 0
                print(faceimg.shape[0], faceimg.shape[1])
                if (faceimg.shape[0] < 80 and faceimg.shape[1] < 80):
                    faceimg, _ = self.srgan.LR_to_HR(faceimg)
                    cv2.imwrite('samples/Face' + str(i)+'.png', faceimg)
                    flag = 1
                if faceimg.ndim < 2:
                    print('Unable to align ')
                faces = self.face_recognition.identify(faceimg)
                if (len(faces) > 0):
                    if (faces[0].confidence < thed):
                        faces[0].name = 'unknown'
                    faceinfo = [pid, faces[0].name, flag]
                else:
                    faceinfo = [pid, 'unable', flag]
                print(faceinfo)
                dict_face[id].append(faceinfo)
            self.facelock.acquire()
            self.faceq.put(dict_face)
            self.facelock.release()
            print(' img cost time:%f' % (time.time() - item_tm))

    def run(self):
        self.face_recognition = face.Recognition()
        #self.srgan = LHSRGAN()
        self.srgan = LHFSRGAN()
        self.q.put(os.getpid())
        while self.thread_state:

            msg = self.yoloq.get()
            #print(len(msg))
            print("FaceRecognition~~~~~~~~~~~~~~~~~~~~~%d" % (self.yoloq.qsize()))
            self.yoloMsgProc(msg)
            #self.yoloq.task_done()

            #print(recv)











