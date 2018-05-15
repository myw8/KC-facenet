import numpy as np

def json2data(id,num,xs,ys,ws,hs,ds):
    data = chr(170)
    data += id
    data += chr(num)
    #id num len x y w h img_data len x y w h img_data ...
    for i in range(num):
        d_len = 0
        if(ds != None):
            d_len = len(ds[i])
        d_l = d_len % 256
        d_h = (d_len / 256) % 256
        data += chr(d_l) + chr(d_h)

        d_l = xs[i]%256
        d_h = (xs[i]/256)%256
        data += chr(d_l) + chr(d_h)

        d_l = ys[i] % 256
        d_h = (ys[i] / 256) % 256
        data += chr(d_l) + chr(d_h)

        d_l = ws[i] % 256
        d_h = (ws[i] / 256) % 256
        data += chr(d_l) + chr(d_h)

        d_l = hs[i] % 256
        d_h = (hs[i] / 256) % 256
        data += chr(d_l) + chr(d_h)
        if (ds != None):
            for j in range(len(ds[i])):
                data += chr(ds[i][j])
    data += chr(85)
    return data

def data2json(data):
    # print(data)
    DATA_ID = 23
    DATA_NUM = DATA_ID+1

    DATA_LEN = 0
    DATA_X = 2
    DATA_Y = 4
    DATA_W = 6
    DATA_H = 8
    DATA_D = 10

    xs = []
    ys = []
    ws = []
    hs = []
    ds = []

    start = ord(data[0])
    if(start != 170):
        print('error:', 'start', start)
        return 0, 0, xs, ys, ws, hs, ds
    # print(start)
    id = data[1:DATA_ID+1]
    # print(id)
    num = ord(data[DATA_NUM])
    # print(num)

    nid = DATA_NUM+1
    for i in range(num):
        d_l = ord(data[nid+DATA_LEN])
        d_h = ord(data[nid+DATA_LEN+1])

        n = d_l + d_h*256
        # print(n)
        d_l = ord(data[nid + DATA_X])
        d_h = ord(data[nid + DATA_X + 1])
        d = d_l + d_h * 256
        xs.append(d)
        d_l = ord(data[nid + DATA_Y])
        d_h = ord(data[nid + DATA_Y + 1])
        d = d_l + d_h * 256
        ys.append(d)
        d_l = ord(data[nid + DATA_W])
        d_h = ord(data[nid + DATA_W + 1])
        d = d_l + d_h * 256
        ws.append(d)
        d_l = ord(data[nid + DATA_H])
        d_h = ord(data[nid + DATA_H + 1])
        d = d_l + d_h * 256
        hs.append(d)
        dd = [0 for x in range(0, n)]

        if (n > 0):
            for j in range(n):
                dd[j] = ord(data[nid + DATA_D + j])
            ds.append(dd)
        nid += n + 10
        # print(nid)
    return id,num, xs,ys,ws,hs,ds