#-*-coding:utf-8-*-


import zmq
import numpy as np
import cv2
import base64

def send_msg(socket,image,msg):
    if image.flags['C_CONTIGUOUS']:
        # if image is already contiguous in memory just send it

        w = 200
        h = int(image.shape[0]*w/image.shape[1])

        re_image = cv2.resize(image, (w, h), interpolation = cv2.INTER_CUBIC)

        img_str = cv2.imencode('.jpg', re_image)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
        b64_code = base64.b64encode(img_str)

        md = dict(
            msg=msg,
            dtype=str(image.dtype),
            shape=image.shape,
            data=str(b64_code, encoding = "utf8"),
        )

        socket.send_json(md)
        #socket.send(image, 0, copy=True, track=False)
    else:
        # else make it contiguous before sending
        image = np.ascontiguousarray(image)

        w = 200
        h = int(image.shape[0]*w/image.shape[1])

        re_image = cv2.resize(image, (w, h), interpolation = cv2.INTER_CUBIC)

        img_str = cv2.imencode('.jpg', re_image)[1].tostring()  # 将图片编码成流数据，放到内存缓存中，然后转化成string格式
        b64_code = base64.b64encode(img_str)

        md = dict(
            msg=msg,
            dtype=str(image.dtype),
            shape=image.shape,
            data=str(b64_code, encoding = "utf8"),
        )
        socket.send_json(md)
        #socket.send(image, 0, copy=True, track=False)

