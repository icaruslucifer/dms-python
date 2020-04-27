# _*_ coding=UTF-8 _*_

import sys,os,cv2,time
import  numpy as np
from argparse import  ArgumentParser

from openvino_util import  *

from image_util import *

import threading
import area_util

try:
    from tkinter import *
    from tkinter import filedialog
except:
    from Tkinter import *


landmark_input_size = 60


face_input_width = 672
face_input_height = 384


label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
landmark_color = (0, 0, 255)
box_thickness = 1

cur_thread = None
video_fps = 20
fps_label = None


labels_imgaepaths = ['./data/eyes/left','./data/eyes/right','./data/mouth','./data/ears/left','./data/ears/right']
labels = ['左眼','右眼','嘴部','左耳','右耳']
labels_value = []

need_label = False
pos_Var = None


for p in labels_imgaepaths:
    if not os.path.exists(os.path.join(p,"neg")):
        os.makedirs(os.path.join(p,"neg"),mode=0o777)
    if not os.path.exists(os.path.join(p,"pos")):
        os.makedirs(os.path.join(p,"pos"),mode=0o777)




def choose_video_click():
    filepath = filedialog.askopenfilename()
    if len(filepath) >0 :
        print(filepath)
        global cur_thread
        if cur_thread != None:
            try:
                stop_thread(cur_thread)
                cur_thread = None
            except:
                pass
        cur_thread = MyVideoThread(filepath)
        cur_thread.start()

    else:
        print('error path')


import ctypes
import inspect
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


class MyVideoThread(threading.Thread):
    def __init__(self,videopath):
        super(MyVideoThread,self).__init__()
        self.videopath = videopath

    def run(self):
        cap = cv2.VideoCapture(self.videopath)
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
                    pass
                else:
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

                        if need_label:
                            label_image(oriimage,t_landmarks)
                    if need_label:
                        need_label = False
                t = 1000/video_fps
                # cost = time.clock()-a_start
                # t -= cost*1000
                if t <= 0:
                    t = 1
                k = cv2.waitKey(int(t)) & 0xFF
                if k == ord('q'):
                    cv2.destroyAllWindows()
                    break
                elif k == ord(' '):
                    cv2.waitKey(0)
                elif k == ord('s'):
                    bbox = cv2.selectROI(oriimage)
                    if bbox[2]>0:
                        cv2.imwrite("./data/"+str(time.time())+".png", oriimage[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]])
                try:
                    if image.shape[0] > 0:
                        cv2.namedWindow("video", 0)
                        cv2.resizeWindow("video", 400, image.shape[0]*400//image.shape[1])
                        cv2.imshow("video", image)
                except:
                    pass
            except:
                pass
        cv2.destroyAllWindows()



def label_image(image,landmarks):
    try:
        for i,e in enumerate(labels_value):
            if e.get() == 1:
                print(labels[i])
                if pos_Var.get() == 1:
                    fpath = os.path.join(labels_imgaepaths[i],"pos","pos"+str(i)+"_"+str(time.time()))+".png"
                else:
                    fpath = os.path.join(labels_imgaepaths[i],"neg","neg"+str(i)+"_"+str(time.time()))+".png"
                if i == 2:
                    img = area_util.getSmokeImage(image,landmarks)
                    img = cv2.resize(img, (mouth_width, mouth_height))
                    cv2.imwrite(fpath, img)
                elif i == 3:
                    img = area_util.getLeftPhoneImage(image,landmarks)
                    img = cv2.resize(img, (ear_width, ear_height))
                    cv2.imwrite(fpath, img)
                elif i == 4:
                    img = area_util.getRightPhoneImage(image,landmarks)
                    img = cv2.resize(img, (ear_width, ear_height))
                    cv2.imwrite(fpath, img)
    except:
        pass


def update_fps_label():
    global fps_label
    t = "播放帧率:"+str(video_fps)
    fps_label['text'] = t


def add_fps_click():
    global video_fps,fps_label
    video_fps +=1
    update_fps_label()

def minus_fps_click():
    global video_fps,fps_label
    video_fps -=1
    if video_fps <2:
        video_fps = 1
    update_fps_label()

def label_click():
    global need_label
    need_label = True

def main():
    main_window = Tk()
    main_window.title("DMS样本标签采集")
    window_width = 300
    window_height = 300

    screen_width = main_window.winfo_screenwidth()
    screen_height = main_window.winfo_screenheight()

    alignstr = '%dx%d+%d+%d' % (window_width,window_height,(screen_width-window_width)/2,(screen_height-window_height)/2)
    main_window.geometry(alignstr)

    frame_main = Frame(main_window,borderwidth=1,relief=SUNKEN)
    frame_main.place(x = 0,y=0,width = window_width,height = window_height)
    choose_video_button = Button(frame_main,text ="选择视频文件", command = choose_video_click)
    choose_video_button.grid(row = 0 ,column=0)

    global fps_label
    fps_label = Label(frame_main, text="播放帧率:"+str(video_fps), font=("Arial", 12), width=10, height=2)
    fps_label.grid(row=1,column=0)
    add_fps_button = Button(frame_main,text =" + ", command = add_fps_click)
    add_fps_button.grid(row = 1 ,column=1)
    minus_fps_button = Button(frame_main,text =" - ", command = minus_fps_click)
    minus_fps_button.grid(row = 1 ,column=2)

    global pos_Var
    pos_Var = IntVar()
    Checkbutton(frame_main,text="正样本",variable = pos_Var).grid(row=2,column=0)

    label_button = Button(frame_main,text ="抠样本", command = label_click)
    label_button.grid(row = 2 ,column=0,rowspan=len(labels))
    global labels_value
    for i in range(len(labels)):
        labels_value.append(IntVar())
        Checkbutton(frame_main,text=labels[i],variable = labels_value[-1]).grid(row=3+i,column=1)

    main_window.mainloop()


if __name__ == '__main__':
    main()