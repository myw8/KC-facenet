# -*- coding: utf8 -*-
# ! /usr/bin/python
import sys,signal
from recognition_process.face_subscriber_thread import FaceSubscriber
from recognition_process.face_publisher_thread import FacePublisher
from recognition_process.face_recognition_process import FaceRecognition
from multiprocessing import Queue,Lock
import time
def sigint_handler(signum,frame):
    print("main-thread exit")
    global subs, pubs, face_proc
    pubs.stop()
    subs.stop()
    for i in range(len(face_proc)):
        face_proc[i].stop()

    sys.exit()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    thed = 0.6
    procnum = 2
    load_mode_finish_q = Queue()
    face_msg_queue = Queue()
    yolo_msg_queue = Queue()

    subs = FaceSubscriber("tcp://127.0.0.1:12341","tcp://127.0.0.1:12302",yolo_msg_queue)
    subs.setDaemon(True)

    pubs = FacePublisher("tcp://127.0.0.1:12354",face_msg_queue)
    pubs.setDaemon(True)

    yololock = Lock()
    facelock = Lock()
    face_proc = []
    for i in range(procnum):
        face_proc.append(FaceRecognition(load_mode_finish_q,yolo_msg_queue,yololock,face_msg_queue,facelock))
    for i in range(procnum):
        face_proc[i].start()

    while load_mode_finish_q.qsize()<procnum:
        #print(load_mode_finish_q.get())
        time.sleep(1)

    pubs.start()
    subs.start()


    for i in range(len(face_proc)):
        face_proc[i].join()
    subs.join()
    pubs.join()








