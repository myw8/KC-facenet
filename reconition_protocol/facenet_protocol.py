# -*- coding: utf8 -*-
# ! /usr/bin/python
import numpy as np
import cv2
#[START1,START2,LEN,NUM,FRAMEID,[PID,NAME,AIFLAG,h,w,IMG]........END]
class FacenetProtocol(object):
    START1 = 0
    START2 = 1
    LEN = 2
    FRAMID = 6
    FRAMEIDLEN = 23
    NAMElEN = 20
    def __init__(self):
        self.protobuf=None
        #self.protobuf= chr(0xA5)
        #self.protobuf += chr(0x5A)
        self.len = 0
        self.num = 0
        self.frameid = None



    def addFrameId(self,frameId):
        self.frameid = frameId
        self.protobuf = frameId.zfill(self.FRAMEIDLEN)
        print(frameId)
        self.len += self.FRAMEIDLEN+4


    def addFaceMsg(self,pId,name,aiflag,img,h,w):
        self.num += 1

        d_l = pId % 256
        d_h = (pId / 256) % 256
        self.protobuf += chr(d_l) + chr(d_h)
        self.len += 2

        self.protobuf += name.zfill(self.NAMElEN)
        self.len += self.NAMElEN

        self.protobuf += chr(aiflag)
        self.len +=1

        d_l = h % 256
        d_h = (h / 256) % 256
        self.protobuf += chr(d_l) + chr(d_h)
        self.len += 2

        d_l = w % 256
        d_h = (w / 256) % 256
        self.protobuf += chr(d_l) + chr(d_h)
        self.len += 2

        print('send Pid:%d imglen:%f w:%d h:%d'%(pId,len(img),w,h))
        if(w !=0 and h != 0):
            for j in range(len(img)):
               self.protobuf += chr(img[j])
            self.len += len(img)


    def addEnd(self):
        print(self.len)
        d1 = self.len % 256
        d2 = (self.len >>8) % 256
        d3 = (self.len >>16) % 256
        d4 = (self.len >>24) % 256
        num1 = self.num % 256
        num2 = (self.num / 256) % 256
        self.protobuf = chr(0xA5)+chr(0x5A)+chr(d1) + chr(d2)+chr(d3)+chr(d4)+\
                        chr(num1)+chr(num2)+self.protobuf+chr(0x16)

    def getProtoBuf(self):
        return self.protobuf

    def parserFaceData(self,data):
        start1 = ord(data[self.START1])
        start2 = ord(data[self.START2])

        d1 = ord(data[self.LEN])
        d2 = ord(data[self.LEN + 1])
        d3 = ord(data[self.LEN + 2])
        d4 = ord(data[self.LEN + 3])

        n_len = d1 + d2*256 + d3*256*256 + d4*256*256*256
        print(len(data))
        print(n_len+7)
        end = ord(data[len(data)-1])
        if ((start1 != 0xA5) and (start2 != 0x5A) and (end != 0x16)):
            print('error:', 'start', start1,start2,end)
            return None
        offset = 6

        d_l = ord(data[offset ])
        d_h = ord(data[offset  + 1])
        num = d_l + d_h * 256
        offset += 2

        frameId = data[offset:offset + self.FRAMEIDLEN ]
        print(frameId)
        dict_face = {}
        dict_face[frameId] = []
        offset = offset + self.FRAMEIDLEN
        for i in range(num):

            d_l = ord(data[offset])
            d_h = ord(data[offset + 1])
            pid = d_l + d_h * 256
            offset += 2




            name = data[offset:offset + self.NAMElEN]
            offset = offset+self.NAMElEN

            aiflag = ord(data[offset])
            offset += 1

            #img h w

            d_l = ord(data[offset])
            d_h = ord(data[offset + 1])
            h = d_l + d_h * 256
            offset += 2



            d_l = ord(data[offset])
            d_h = ord(data[offset + 1])
            w = d_l + d_h * 256
            offset += 2

            imglen = h * w * 3
            print('recv  Pid:%d img h:%d w:%d' % (pid, h,w))


            img = np.zeros((imglen,), dtype=np.uint8)
            if (imglen > 0):
                for j in range(imglen):
                    img[i] = ord(data[offset + j])
                if 1:
                    cv2.imshow("Face: ", img.reshape(h, w, 3))
            offset += imglen

            print([pid,name.replace('0',''),aiflag,img])
            dict_face[frameId].append([pid,name.replace('0',''),aiflag,img])

        print(dict_face)
        return dict_face


