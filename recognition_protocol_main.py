# -*- coding: utf8 -*-
# ! /usr/bin/python
import sys,signal
from reconition_protocol.face_subscriber_protocol import FaceSubscriber
from reconition_protocol.face_publisher_protocol import FacePublisher
from reconition_protocol.face_recognition_protocol import FaceRecognition
from multiprocessing import Queue,Lock
import time
from  src import classifier
def sigint_handler(signum,frame):
    print("main-thread exit")
    global subs, pubs, face_proc
    pubs.stop()
    subs.stop()
    for i in range(len(face_proc)):
        face_proc[i].stop()

    sys.exit()

def get_classes(path):
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    return classes
def KNN_train(arg):
    classifier.main(arg)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    thed = 0.6
    procnum = 1
    load_mode_finish_q = Queue()
    face_msg_queue = Queue()
    yolo_msg_queue = Queue()

    subs = FaceSubscriber("tcp://127.0.0.1:812341","tcp://127.0.0.1:812302",yolo_msg_queue)
    subs.setDaemon(True)


    pubs = FacePublisher("tcp://127.0.0.1:812354", face_msg_queue)
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
        load_mode_finish_q.get()

    config = edict()
    config.mode = 16
    config.data_dir = 1e-4
    config.classifier = 'KNN'
    config.use_split_dataset = 0
    config.test_data_dir = ''
    config.batch_size = 100
    config.image_size = 160
    config.seed = 666
    config.min_nrof_images_per_class = 5
    config.nrof_train_images_per_class = 5

    knn_classes = get_classes('/opt/yanhong.jia/classes')
    while True:
        time.sleep(60)
        classes = get_classes('/opt/yanhong.jia/classes')
        if(len(classes) != len(knn_classes)):
            #start classifier train
            knn_train = Process(target=KNN_train, args=(config,))
            knn_train.start()
            knn_train.join()

            for i in range(len(face_proc)):
                face_proc[i].stop()
                face_proc[i].start()
                print(load_mode_finish_q.get())



        for i in range(classes):
            if(classes[i] != knn_classes[i]):
                # start classifier train
                break
        knn_train = Process(target=KNN_train, args=(config,))
        knn_train.start()
        knn_train.join()
        for i in range(len(face_proc)):
            face_proc[i].stop()
            face_proc[i].start()
            print(load_mode_finish_q.get())


    for i in range(len(face_proc)):
        face_proc[i].join()
    subs.join()
    pubs.join()










