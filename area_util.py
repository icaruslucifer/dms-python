# _*_ coding=UTF-8 _*_

import sys,os,cv2,time
import math
import  numpy as np


import icarus_point

mounth_scale = 2.0

ear_wh_scale = 1.0


def getRightPhoneDetectArea(landmarks):
    h = ((landmarks[16][1] - landmarks[26][1])**2 + (landmarks[16][0] - landmarks[26][0])**2)**0.5
    w = h * ear_wh_scale

    x1 = float(landmarks[16][0])
    y1 = float(landmarks[16][1])
    x2 = float(landmarks[26][0])
    y2 = float(landmarks[26][1])

    x3 = float(landmarks[3][0])
    y3 = float(landmarks[3][1])



    if landmarks[16][0] == landmarks[26][0] :
        return [landmarks[3][0],landmarks[16][1],
                int(landmarks[3][0]-w),landmarks[16][1],
                int(landmarks[3][0]-w),landmarks[26][1],
                landmarks[3][0],landmarks[26][1]]
    else:

        k = (y2-y1)/(x2-x1)
        if k == 0:
            return [landmarks[3][0],landmarks[16][1],
                    int(landmarks[3][0]-w),landmarks[16][1],
                    int(landmarks[3][0]-w),landmarks[26][1],
                    landmarks[3][0],landmarks[26][1]]
        q = math.atan(abs(k))

        m = w/math.sin(q)

        #l1  k(x-x3)+y3 = y
        l1 = icarus_point.Line(icarus_point.Point(0,y3-k*x3),icarus_point.Point(-y3/k+x3,0))

        #l2 -(x-x1)/k + y1 = y
        l2 = icarus_point.Line(icarus_point.Point(0,y1+x1/k),icarus_point.Point(y1*k+x1,0))

        #l3 k(x-x3 - m)+y3 = y
        l3 = icarus_point.Line(icarus_point.Point(0,y3+k*(-m-x3)),icarus_point.Point(-y3/k+x3+m,0))

        #l4 -(x-x2)/k + y2 = y
        l4 = icarus_point.Line(icarus_point.Point(0,y2+x2/k),icarus_point.Point(y2*k+x2,0))

        k1 = icarus_point.GetCrossPoint(l1,l2)
        k2 = icarus_point.GetCrossPoint(l2,l3)
        k3 = icarus_point.GetCrossPoint(l3,l4)
        k4 = icarus_point.GetCrossPoint(l4,l1)

        return [int(k1.x),int(k1.y),int(k2.x),int(k2.y),int(k3.x),int(k3.y),int(k4.x),int(k4.y)]

def getRightPhoneImage(image,landmarks):
    ph = ((landmarks[16][1] - landmarks[26][1])**2 + (landmarks[16][0] - landmarks[26][0])**2)**0.5
    pw = ph * ear_wh_scale


    x1 = float(landmarks[16][0])
    y1 = float(landmarks[16][1])
    x2 = float(landmarks[26][0])
    y2 = float(landmarks[26][1])

    if landmarks[16][0] == landmarks[26][0]:
        return image[landmarks[16][1]:landmarks[26][1],int(landmarks[3][0]):int(landmarks[3][0]+pw)]
    else:
        k = (y2-y1)/(x2-x1)

        if k == 0:
            return image[landmarks[16][1]:landmarks[26][1],int(landmarks[3][0]):int(landmarks[3][0]+pw)]

        q = math.atan(k)

        x3 = float(landmarks[3][0])
        y3 = float(landmarks[3][1])

        #l1  k(x-x3)+y3 = y
        l1 = icarus_point.Line(icarus_point.Point(0,y3-k*x3),icarus_point.Point(-y3/k+x3,0))

        #l2 -(x-x1)/k + y1 = y
        l2 = icarus_point.Line(icarus_point.Point(0,y1+x1/k),icarus_point.Point(y1*k+x1,0))

        k1 = icarus_point.GetCrossPoint(l1,l2)

        (h, w) = image.shape[:2]
        if k>0:
            M = cv2.getRotationMatrix2D((int(k1.x),int(k1.y)), -90+q*180/math.pi, 1.0)
        else:
            M = cv2.getRotationMatrix2D((int(k1.x),int(k1.y)), 90+q*180/math.pi, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated[int(k1.y):int(k1.y+ph),int(k1.x):int(k1.x+pw)]


def getLeftPhoneDetectArea(landmarks):
    h = ((landmarks[13][1] - landmarks[26][1])**2 + (landmarks[13][0] - landmarks[26][0])**2)**0.5
    w = h * ear_wh_scale

    x1 = float(landmarks[13][0])
    y1 = float(landmarks[13][1])
    x2 = float(landmarks[26][0])
    y2 = float(landmarks[26][1])

    x3 = float(landmarks[1][0])
    y3 = float(landmarks[1][1])


    if landmarks[13][0] == landmarks[26][0] :
        return [landmarks[1][0],landmarks[13][1],
                int(landmarks[1][0]-w),landmarks[13][1],
                int(landmarks[1][0]-w),landmarks[26][1],
                landmarks[1][0],landmarks[26][1]]
    else:

        k = (y2-y1)/(x2-x1)
        if k == 0:
            return [landmarks[1][0],landmarks[13][1],
                    int(landmarks[1][0]-w),landmarks[13][1],
                    int(landmarks[1][0]-w),landmarks[26][1],
                    landmarks[1][0],landmarks[26][1]]
        q = math.atan(abs(k))

        m = w/math.sin(q)

        #l1  k(x-x3)+y3 = y
        l1 = icarus_point.Line(icarus_point.Point(0,y3-k*x3),icarus_point.Point(-y3/k+x3,0))

        #l2 -(x-x1)/k + y1 = y
        l2 = icarus_point.Line(icarus_point.Point(0,y1+x1/k),icarus_point.Point(y1*k+x1,0))

        #l3 k(x-x3 +m)+y3 = y
        l3 = icarus_point.Line(icarus_point.Point(0,y3+k*(m-x3)),icarus_point.Point(-y3/k+x3-m,0))

        #l4 -(x-x2)/k + y2 = y
        l4 = icarus_point.Line(icarus_point.Point(0,y2+x2/k),icarus_point.Point(y2*k+x2,0))

        k1 = icarus_point.GetCrossPoint(l1,l2)
        k2 = icarus_point.GetCrossPoint(l2,l3)
        k3 = icarus_point.GetCrossPoint(l3,l4)
        k4 = icarus_point.GetCrossPoint(l4,l1)

        return [int(k1.x),int(k1.y),int(k2.x),int(k2.y),int(k3.x),int(k3.y),int(k4.x),int(k4.y)]

def getLeftPhoneImage(image,landmarks):
    ph = ((landmarks[13][1] - landmarks[26][1])**2 + (landmarks[13][0] - landmarks[26][0])**2)**0.5
    pw = ph * ear_wh_scale


    x1 = float(landmarks[13][0])
    y1 = float(landmarks[13][1])
    x2 = float(landmarks[26][0])
    y2 = float(landmarks[26][1])

    if landmarks[13][0] == landmarks[26][0]:
        return image[landmarks[13][1]:landmarks[26][1],int(landmarks[1][0]-pw):landmarks[1][0]]
    else:
        k = (y2-y1)/(x2-x1)
        q = math.atan(k)
        if k == 0:
            return image[landmarks[13][1]:landmarks[26][1],int(landmarks[1][0]-pw):landmarks[1][0]]
        x3 = float(landmarks[1][0])
        y3 = float(landmarks[1][1])

        #l1  k(x-x3)+y3 = y
        l1 = icarus_point.Line(icarus_point.Point(0,y3-k*x3),icarus_point.Point(-y3/k+x3,0))

        #l2 -(x-x1)/k + y1 = y
        l2 = icarus_point.Line(icarus_point.Point(0,y1+x1/k),icarus_point.Point(y1*k+x1,0))

        k1 = icarus_point.GetCrossPoint(l1,l2)

        (h, w) = image.shape[:2]
        if k>0:
            M = cv2.getRotationMatrix2D((int(k1.x),int(k1.y)), -90+q*180/math.pi, 1.0)
        else:
            M = cv2.getRotationMatrix2D((int(k1.x),int(k1.y)), 90+q*180/math.pi, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated[int(k1.y):int(k1.y+ph),int(k1.x-pw):int(k1.x)]

def getEyesArea(landmarks):
    #眼睛范围取两眼正中心之间的距离 的 5/3 和10/9
    x1 = (landmarks[0][0] +landmarks[1][0])/2.0
    y1 = (landmarks[0][1] +landmarks[1][1])/2.0
    x2 = (landmarks[2][0] +landmarks[3][0])/2.0
    y2 = (landmarks[2][1] +landmarks[3][1])/2.0

    cx = (x1+x2)/2
    cy = (y1+y2)/2

    d = ((x1-x2)**2+(y1-y2)**2)**0.5
    w = d*5/3.0
    h = d*10/9.0

    if y1 == y2:
        l = (x1+x2)/2 - w/2
        t = y1-h/2
        return [int(l),int(t),int(l+w),int(t),int(l+w),int(t+h),int(l),int(t+h)]
    elif x1 == x2:
        t = (y1+y2)/2 - w/2
        l = x1-h/2
        return [int(l+w),int(t),int(l+w),int(t+h),int(l),int(t+h),int(l),int(t)]
    else:
        k = (y1-y2)/(x1-x2)

        q = math.atan(abs((y2-y1)/(x2-x1)))
        m = h/2/math.sin(q)
        n = w/2/math.cos(q)

        if k >0:


            #l1  k(x-cx+m)+cy = y
            l1 = icarus_point.Line(icarus_point.Point(0,k*(-cx+m)+cy),icarus_point.Point(-cy/k+cx-m,0))

            #l2  -(x-cx-n)/k+cy = y
            l2 = icarus_point.Line(icarus_point.Point(0,-(-cx-n)/k+cy),icarus_point.Point(cy*k+cx+n,0))

            #l3  k(x-cx-m)+cy = y
            l3 = icarus_point.Line(icarus_point.Point(0,k*(-cx-m)+cy),icarus_point.Point(-cy/k+cx+m,0))

            #l4  -(x-cx+n)/k+cy = y
            l4 = icarus_point.Line(icarus_point.Point(0,-(-cx+n)/k+cy),icarus_point.Point(cy*k+cx-n,0))

            k1 = icarus_point.GetCrossPoint(l1,l2)
            k2 = icarus_point.GetCrossPoint(l2,l3)
            k3 = icarus_point.GetCrossPoint(l3,l4)
            k4 = icarus_point.GetCrossPoint(l4,l1)

            return [int(k1.x),int(k1.y),int(k2.x),int(k2.y),int(k3.x),int(k3.y),int(k4.x),int(k4.y)]
        else:
            #l1  k(x-cx+h/2)+cy = y
            l3 = icarus_point.Line(icarus_point.Point(0,k*(-cx+m)+cy),icarus_point.Point(-cy/k+cx-m,0))

            #l2  -(x-cx-w/2)/k+cy = y
            l4 = icarus_point.Line(icarus_point.Point(0,-(-cx-n)/k+cy),icarus_point.Point(cy*k+cx+n,0))

            #l3  k(x-cx-h/2)+cy = y
            l1 = icarus_point.Line(icarus_point.Point(0,k*(-cx-m)+cy),icarus_point.Point(-cy/k+cx+m,0))

            #l4  -(x-cx+w/2)/k+cy = y
            l2 = icarus_point.Line(icarus_point.Point(0,-(-cx+n)/k+cy),icarus_point.Point(cy*k+cx-n,0))

            k1 = icarus_point.GetCrossPoint(l1,l2)
            k2 = icarus_point.GetCrossPoint(l2,l3)
            k3 = icarus_point.GetCrossPoint(l3,l4)
            k4 = icarus_point.GetCrossPoint(l4,l1)

            return [int(k1.x),int(k1.y),int(k2.x),int(k2.y),int(k3.x),int(k3.y),int(k4.x),int(k4.y)]


def getEyesImage(image,landmarks):

    x1 = (landmarks[0][0] +landmarks[1][0])/2.0
    y1 = (landmarks[0][1] +landmarks[1][1])/2.0
    x2 = (landmarks[2][0] +landmarks[3][0])/2.0
    y2 = (landmarks[2][1] +landmarks[3][1])/2.0

    cx = (x1+x2)/2
    cy = (y1+y2)/2

    d = ((x1-x2)**2+(y1-y2)**2)**0.5
    w = d*5/3.0
    h = d*10/9.0

    if x1 ==x2:
        return image[int(cy-w/2):int(cy+w/2),int(cx-h/2):int(cx+h/2)]
    else:
        q = math.atan((y2-y1)/(x2-x1))

        (ih, iw) = image.shape[:2]

        # 将图像旋转180度
        M = cv2.getRotationMatrix2D((cx,cy), q*180/math.pi, 1.0)
        rotated = cv2.warpAffine(image, M, (iw, ih))
        return rotated[int(cy-h/2):int(cy+h/2),int(cx-w/2):int(cx+w/2)]


def getSmokeDetectArea(landmarks):
    w = abs(landmarks[8][0] -landmarks[9][0])
    w *= mounth_scale
    cx = landmarks[8][0]/2 +landmarks[9][0]/2
    cy = landmarks[8][1]/2 +landmarks[9][1]/2

    x1 = float(landmarks[8][0])
    y1 = float(landmarks[8][1])
    x2 = float(landmarks[9][0])
    y2 = float(landmarks[9][1])

    if w == 0:
        w = abs(landmarks[8][1] -landmarks[9][1])*mounth_scale
        return [int(cx-w/2),int(cy-w/2),
                int(cx-w/2),int(cy+w/2),
                int(cx+w/2),int(cy+w/2),
                int(cx+w/2),int(cy-w/2)]
    elif landmarks[8][1] -landmarks[9][1] == 0:
        return [int(cx-w/2),int(cy-w/2),
                int(cx-w/2),int(cy+w/2),
                int(cx+w/2),int(cy+w/2),
                int(cx+w/2),int(cy-w/2)]


    q = math.atan(abs((y2-y1)/(x2-x1)))
    r = w/2
    m = r/math.sin(q)
    n = r/math.cos(q)

    #l1  (y2-y1)(x-x1+m)/(x2-x1)+y1 = y
    l1 = icarus_point.Line(icarus_point.Point(0,(y2-y1)*(-x1+m)/(x2-x1)+y1),icarus_point.Point(-y1*(x2-x1)/(y2-y1)+x1-m,0))

    # l2 -(x2-x1)(x-(x1+x2)/2+n)/(y2-y1)+(y1+y2)/2 = y
    l2 = icarus_point.Line(icarus_point.Point(0,-(x2-x1)*(n-(x1+x2)/2)/(y2-y1)+(y1+y2)/2),icarus_point.Point((y1+y2)*(y2-y1)/(x2-x1)/2+(x1+x2)/2-n,0))

    # l3 (y2-y1)(x-x1-m)/(x2-x1)+y1 = y
    l3 = icarus_point.Line(icarus_point.Point(0,(y2-y1)*(-x1-m)/(x2-x1)+y1),icarus_point.Point(-y1*(x2-x1)/(y2-y1)+x1+m,0))

    # l4 -(x2-x1)(x-(x1+x2)/2-n)/(y2-y1)+(y1+y2)/2 = y
    l4 = icarus_point.Line(icarus_point.Point(0,-(x2-x1)*(-(x1+x2)/2-n)/(y2-y1)+(y1+y2)/2),icarus_point.Point((y1+y2)*(y2-y1)/(x2-x1)/2+(x1+x2)/2+n,0))

    k1 = icarus_point.GetCrossPoint(l1,l2)
    k2 = icarus_point.GetCrossPoint(l2,l3)
    k3 = icarus_point.GetCrossPoint(l3,l4)
    k4 = icarus_point.GetCrossPoint(l4,l1)

    return [int(k1.x),int(k1.y),int(k2.x),int(k2.y),int(k3.x),int(k3.y),int(k4.x),int(k4.y)]


def getSmokeImage(image,landmarks):
    r = abs(landmarks[8][0] -landmarks[9][0])
    r *= mounth_scale
    cx = landmarks[8][0]/2 +landmarks[9][0]/2
    cy = landmarks[8][1]/2 +landmarks[9][1]/2

    x1 = float(landmarks[8][0])
    y1 = float(landmarks[8][1])
    x2 = float(landmarks[9][0])
    y2 = float(landmarks[9][1])


    if landmarks[8][0] == landmarks[9][0]:
        return image[int(cy-r/2):int(cy+r/2),int(cx-r/2):int(cx+r/2)]

    q = math.atan((y2-y1)/(x2-x1))

    (h, w) = image.shape[:2]

    # 将图像旋转180度
    M = cv2.getRotationMatrix2D((cx,cy), q*180/math.pi, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated[int(cy-r/2):int(cy+r/2),int(cx-r/2):int(cx+r/2)]
