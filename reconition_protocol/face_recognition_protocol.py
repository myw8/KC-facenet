# -*- coding: utf8 -*-
# ! /usr/bin/python
from FSRGAN.FSRGAN import LHFSRGAN
from multiprocessing import Process
from src import face

import numpy as np
import cv2
import time
import os
import com
from reconition_protocol.facenet_protocol import FacenetProtocol

class FaceRecognition(Process):
#class FaceRecognition(threading.Thread):
    def __init__(self,q,yoloq,yololock,faceq,facelock):
        Process.__init__(self)
        #threading.Thread.__init__(self)
        self.thread_state = False
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

        print(len(msg))
        id, num, xs, ys, ws, hs, ds = com.data2json(msg)
        facemsg = FacenetProtocol()
        facemsg.addFrameId(id)
        start_time = time.time()
        for i in range(num):
            pid = i
            data = ds[i]
            arr = np.array(data)
            arr = arr.reshape(-1)
            try:
                img = arr.reshape(hs[i], ws[i], 3)
            except:
                continue
            print(img.shape)
            img = np.uint8(img)
            faceimg = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
            if debug:
                cv2.imshow("Face: " + str(i), img)
            # face rgb
            #########################################
            flag = 0
            if (i < 2):
                facemsg.addFaceMsg(pid, 'yanhong.jia', flag, [], 0, 0)
            else:
                # print(faceimg.shape,faceimg.reshape(-1).shape,hs[i]*ws[i]*3)
                facemsg.addFaceMsg(pid, 'yanhong.jia', flag, faceimg.reshape(-1), hs[i], ws[i])

        facemsg.addEnd()
        self.facelock.acquire()
        self.faceq.put(facemsg.getProtoBuf())
        self.facelock.release()
        print(' img cost time:%f' % (time.time() - start_time))

    def run(self):
        self.face_recognition = face.Recognition()
        #self.srgan = LHSRGAN()
        self.srgan = LHFSRGAN()
        self.q.put(os.getpid())
        self.thread_state = True
        while self.thread_state:
            msg = self.yoloq.get()
            #print(len(msg))
            print("FaceRecognition~~~~~~~~~~~~~~~~~~~~~%d" % (self.yoloq.qsize()))
            self.yoloMsgProc(msg)
            #self.yoloq.task_done()

            #print(recv)











