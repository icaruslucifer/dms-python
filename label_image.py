#-*-coding:utf-8-*-


import sys,os,cv2,time
import  numpy as np
from argparse import  ArgumentParser

from openvino_util import  *

import threading
import area_util
import glob

import random

from image_util import  *


landmark_input_size = 60


face_input_width = 672
face_input_height = 384


label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
landmark_color = (0, 0, 255)
box_thickness = 1

wait_time = 1

videos = glob.glob("./videos/video/509/*.avi")


cur_index = 2
labels_imgaepaths = ['./data//image/eyes','./data/image/mouth_add','./data//image/ears_add']



for p in labels_imgaepaths:
    if not os.path.exists(os.path.join(p,"neg")):
        os.makedirs(os.path.join(p,"neg"),mode=0o777)
    if not os.path.exists(os.path.join(p,"pos")):
        os.makedirs(os.path.join(p,"pos"),mode=0o777)



def label_image(image,isPos):
    try:
        fpath = labels_imgaepaths[cur_index]

        s = "pos" if isPos else "neg"
        fpath = os.path.join(fpath,s,s+"_"+str(cur_index)+"_"+str(time.time())+".png")
        cv2.imwrite(fpath, image)
    except:
        pass

def label_image_runloop():
    need_detect = True
    for videopath in videos:
        cap = cv2.VideoCapture(videopath)
        ret, image = cap.read()
        if not ret:
            print ("cannot read image")
            sys.exit()

        if float(image.shape[1]) / image.shape[0] < float(face_input_width) / face_input_height:
            new_w = image.shape[1] * face_input_height // image.shape[0]
            new_h = face_input_height
        else:
            new_w = face_input_width
            new_h = image.shape[0] * face_input_width // image.shape[1]

        face_input_v ,face_exec_net,face_plugin = loadfacemodel("CPU")
        landmark_input_v,landmark_exec_net,landmark_plugin = loadfacelandmarkmodel("CPU")


        while cap.isOpened():
            try:
                ret, oriimage = cap.read()
                if not ret:
                    print ("cannot read image")
                    cv2.destroyAllWindows()
                    break
                else:
                    if need_detect:
                        global need_label
                        image = oriimage.copy()
                        a_start = time.clock()
                        resized_image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
                        canvas = np.full((face_input_height, face_input_width, 3), 128)
                        canvas[(face_input_height-new_h)//2:(face_input_height-new_h)//2 + new_h,(face_input_width-new_w)//2:(face_input_width-new_w)//2 + new_w,  :] = resized_image
                        prepimg = canvas
                        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
                        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
                        start =time.clock()
                        outputs = face_exec_net.infer(inputs={face_input_v: prepimg})
                        #print('face detect time:'+str(time.clock()-start))
                        faces = parseFaceOutput(outputs,new_h,new_w,image.shape[0],image.shape[1],0.7)

                        for face in faces:
                            face_image = image[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
                            cv2.rectangle(image, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), box_color, box_thickness)
                            # 检测人脸
                            f_size = face_image.shape[0]
                            if face_image.shape[0]<face_image.shape[1]:
                                f_size = face_image.shape[1]
                            canvas = np.full((f_size, f_size, 3), 128).astype(np.uint8)
                            canvas[(f_size-face_image.shape[0])//2:(f_size-face_image.shape[0])//2 + face_image.shape[0],(f_size-face_image.shape[1])//2:(f_size-face_image.shape[1])//2 + face_image.shape[1],  :] = face_image
                            resized_image = cv2.resize(canvas, (landmark_input_size, landmark_input_size), interpolation = cv2.INTER_CUBIC)
                            prepimg = resized_image
                            prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
                            prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
                            start =time.clock()
                            outputs = landmark_exec_net.infer(inputs={landmark_input_v: prepimg})
                            #print('landmark detect time:'+str(time.clock()-start))
                            landmarks = parseLandmarkOutput(outputs,face_image.shape[0],face_image.shape[1])
                            t_landmarks = []
                            for landmark in landmarks:
                                x = landmark[0]+ face[0]
                                y = landmark[1]+ face[1]
                                t_landmarks.append([x,y])
                                cv2.circle(image,(x,y),2,landmark_color, box_thickness)
                            # 检查打电话
                            leftphone_detectarea = area_util.getLeftPhoneDetectArea(t_landmarks)
                            cv2.line(image, (leftphone_detectarea[0], leftphone_detectarea[1]), (leftphone_detectarea[2], leftphone_detectarea[3]), box_color, box_thickness)
                            cv2.line(image, (leftphone_detectarea[2], leftphone_detectarea[3]), (leftphone_detectarea[4], leftphone_detectarea[5]), box_color, box_thickness)
                            cv2.line(image, (leftphone_detectarea[4], leftphone_detectarea[5]), (leftphone_detectarea[6], leftphone_detectarea[7]), box_color, box_thickness)
                            cv2.line(image, (leftphone_detectarea[6], leftphone_detectarea[7]), (leftphone_detectarea[0], leftphone_detectarea[1]), box_color, box_thickness)

                            rightphone_detectarea = area_util.getRightPhoneDetectArea(t_landmarks)
                            cv2.line(image, (rightphone_detectarea[0], rightphone_detectarea[1]), (rightphone_detectarea[2], rightphone_detectarea[3]), box_color, box_thickness)
                            cv2.line(image, (rightphone_detectarea[2], rightphone_detectarea[3]), (rightphone_detectarea[4], rightphone_detectarea[5]), box_color, box_thickness)
                            cv2.line(image, (rightphone_detectarea[4], rightphone_detectarea[5]), (rightphone_detectarea[6], rightphone_detectarea[7]), box_color, box_thickness)
                            cv2.line(image, (rightphone_detectarea[6], rightphone_detectarea[7]), (rightphone_detectarea[0], rightphone_detectarea[1]), box_color, box_thickness)

                            # 抽烟区域
                            smokedetectArea = area_util.getSmokeDetectArea(t_landmarks)
                            cv2.line(image, (smokedetectArea[0], smokedetectArea[1]), (smokedetectArea[2], smokedetectArea[3]), (255,255,0), box_thickness)
                            cv2.line(image, (smokedetectArea[2], smokedetectArea[3]), (smokedetectArea[4], smokedetectArea[5]), (255,255,0), box_thickness)
                            cv2.line(image, (smokedetectArea[4], smokedetectArea[5]), (smokedetectArea[6], smokedetectArea[7]), (255,255,0), box_thickness)
                            cv2.line(image, (smokedetectArea[6], smokedetectArea[7]), (smokedetectArea[0], smokedetectArea[1]), (255,255,0), box_thickness)
                    else:
                        pass
                global  wait_time
                k = cv2.waitKey(wait_time) & 0xFF
                if k == ord('q'):
                    cv2.destroyAllWindows()
                    break
                elif k == ord(' '):
                    cv2.waitKey(0)
                elif k == ord("s"):
                    label_image(oriimage,True)
                elif k == ord("a"):
                    label_image(oriimage,False)
                elif k == ord("w"):
                    wait_time = 30
                elif k == ord("e"):
                    wait_time = 1
                elif k == ord("d"):
                    need_detect = True
                elif k == ord("n"):
                    need_detect = False
                try:
                    if image.shape[0] > 0:
                        cv2.namedWindow("video", 0)
                        if need_detect:
                            cv2.resizeWindow("video", 600, image.shape[0]*600//image.shape[1])
                            cv2.imshow("video", image)
                        else:
                            cv2.resizeWindow("video", 600, oriimage.shape[0]*600//oriimage.shape[1])
                            cv2.imshow("video", oriimage)
                except:
                    pass
            except:
                pass
        cv2.destroyAllWindows()


def crop_label_image_runloop(path,dirpath=False,needMouth=False,needEar=False,needEye=False,randommouth=False):
    imagepaths = glob.glob(os.path.join(path,"*.png"))
    for index,imagepath in enumerate(imagepaths):
        fpath,tname = os.path.split(imagepath)
        shotname,extension = os.path.splitext(tname)
        image = cv2.imread(imagepath)
        if float(image.shape[1]) / image.shape[0] < float(face_input_width) / face_input_height:
            new_w = image.shape[1] * face_input_height // image.shape[0]
            new_h = face_input_height
        else:
            new_w = face_input_width
            new_h = image.shape[0] * face_input_width // image.shape[1]

        face_input_v ,face_exec_net,face_plugin = loadfacemodel("CPU")
        landmark_input_v,landmark_exec_net,landmark_plugin = loadfacelandmarkmodel("CPU")

        resized_image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
        canvas = np.full((face_input_height, face_input_width, 3), 128)
        canvas[(face_input_height-new_h)//2:(face_input_height-new_h)//2 + new_h,(face_input_width-new_w)//2:(face_input_width-new_w)//2 + new_w,  :] = resized_image
        prepimg = canvas
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        outputs = face_exec_net.infer(inputs={face_input_v: prepimg})
        faces = parseFaceOutput(outputs,new_h,new_w,image.shape[0],image.shape[1],0.7)

        for t,face in enumerate(faces):
            face_image = image[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
            # 检测人脸
            f_size = face_image.shape[0]
            if face_image.shape[0]<face_image.shape[1]:
                f_size = face_image.shape[1]
            canvas = np.full((f_size, f_size, 3), 128).astype(np.uint8)
            canvas[(f_size-face_image.shape[0])//2:(f_size-face_image.shape[0])//2 + face_image.shape[0],(f_size-face_image.shape[1])//2:(f_size-face_image.shape[1])//2 + face_image.shape[1],  :] = face_image
            resized_image = cv2.resize(canvas, (landmark_input_size, landmark_input_size), interpolation = cv2.INTER_CUBIC)
            prepimg = resized_image
            prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
            prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
            start =time.clock()
            outputs = landmark_exec_net.infer(inputs={landmark_input_v: prepimg})
            #print('landmark detect time:'+str(time.clock()-start))
            landmarks = parseLandmarkOutput(outputs,face_image.shape[0],face_image.shape[1])
            t_landmarks = []
            for landmark in landmarks:
                x = landmark[0]+ face[0]
                y = landmark[1]+ face[1]
                t_landmarks.append([x,y])
            # 检查打电话
            if needEar:
                leftphonecall_img = area_util.getLeftPhoneImage(image,t_landmarks)
                if leftphonecall_img.shape[0] != 0 and leftphonecall_img.shape[1] !=0:
                    phone_img1 = cv2.resize(leftphonecall_img, (ear_width, ear_height))
                    cv2.imwrite(os.path.join(dirpath,shotname+"_l.png"),phone_img1)
                rightphonecall_img = area_util.getRightPhoneImage(image,t_landmarks)
                if rightphonecall_img.shape[0] != 0 and rightphonecall_img.shape[1] !=0:
                    phone_img2 = cv2.resize(rightphonecall_img, (ear_width, ear_height))
                    cv2.imwrite(os.path.join(dirpath,shotname+"_r.png"),phone_img2)

            if needMouth:
                smoke_image = area_util.getSmokeImage(image,t_landmarks)
                if smoke_image.shape[0] != 0 and smoke_image.shape[1] !=0:
                    smoke_image = cv2.resize(smoke_image, (mouth_width, mouth_height))
                    cv2.imwrite(os.path.join(dirpath,shotname+"_m"+str(t)+".png"),smoke_image)
                if randommouth:
                    for i in range(3):
                        try:
                            t_image = random_crop_fro_neg(face_image)
                            t_image = cv2.resize(t_image, (mouth_width, mouth_height))
                            cv2.imwrite(os.path.join(dirpath,shotname+"_mr"+str(i)+".png"),t_image)
                        except:
                            pass
        print(str(index)+"-"+str(len(imagepaths)))


def random_crop_fro_neg(image):
    t = image.shape[1]
    if image.shape[0] < image.shape[1]:
        t = image.shape[0]
    r = int(t * 0.5)

    rw = image.shape[1] - r
    rh = image.shape[0] - 0.5*r

    sw = random.randint(0,rw)
    sh = random.randint(int(0.5*r),rh)

    return image[sh:sh+r,sw:sw+r]








def main():

    label_image_runloop()

    # o_path = "./data/image/ears_add/pos"
    # d_path = "./data/image/ears_add/1.0/pos"
    # if not os.path.exists(d_path):
    #     os.makedirs(d_path,mode=0o777)
    # crop_label_image_runloop(o_path,d_path,needEar=True)


if __name__ == '__main__':
    main()


