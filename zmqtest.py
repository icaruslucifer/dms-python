#-*-coding:utf-8-*-


import time
import zmq
import cv2
import base64
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://127.0.0.1:5000")
socket.setsockopt(zmq.SUBSCRIBE,b'')
while True:
    try:
        md = socket.recv_json()
        if "shape" in md:
            str_decode = base64.b64decode(md['data'])
            nparr = np.fromstring(str_decode, np.uint8)
            img_restore = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow("result",img_restore)
            cv2.waitKey(1)
        else:
            print(md)
    except Exception as e:
        pass