# -*- coding: utf8 -*-
# ! /usr/bin/python
import pickle
from facenet_protocol import FacenetProtocol
import com
import numpy as np
import cv2


def yoloMsgProc(msg):
    thed = 0.6
    debug = 0

    print(len(msg))
    id, num, xs, ys, ws, hs, ds = com.data2json(msg)
    facemsg = FacenetProtocol()
    facemsg.addFrameId(id)

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
        if(i<2):
            facemsg.addFaceMsg(pid, 'yanhong.jia', flag,[],0,0)
        else:
            #print(faceimg.shape,faceimg.reshape(-1).shape,hs[i]*ws[i]*3)
            facemsg.addFaceMsg(pid,'yanhong.jia',flag,faceimg.reshape(-1),hs[i], ws[i])

    facemsg.addEnd()
    tmpdata = facemsg.getProtoBuf()
    return tmpdata



#face_msg_queue = Queue()
data= pickle.load(open('yolodata','rb'))
#pubs = FacePublisher("tcp://127.0.0.1:812354", "tcp://127.0.0.1:812301", face_msg_queue)
#pubs.setDaemon(True)
tmpdata = yoloMsgProc(data)
faceProto= FacenetProtocol()
dict_data = faceProto.parserFaceData(tmpdata)
print(dict_data)

