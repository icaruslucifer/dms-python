# _*_ coding=UTF-8 _*_


import sys,os,cv2,time

import threading
lock = threading.RLock()

try:
    from tkinter import *
except:
    from Tkinter import *

main_window = Tk()
main_window.title("DMS视频采集器")
window_width = 300
window_height = 300

screen_width = main_window.winfo_screenwidth()
screen_height = main_window.winfo_screenheight()

alignstr = '%dx%d+%d+%d' % (window_width,window_height,(screen_width-window_width)/2,(screen_height-window_height)/2)
main_window.geometry(alignstr)

camera_id = 0

collect_state = 1 # 1未采集 2 将要采集 3 正在采集 4 结束采集

cur_frame = None

video_out = None
video_save_dir = "./videos"
if not os.path.exists(video_save_dir):
    os.mkdir(video_save_dir)

def open_camera_click():
    pass

def close_camera_click():
    pass


def begin_collect_click():
    global collect_state
    collect_state = 2

    pass

def end_collect_click():
    global collect_state
    collect_state = 4
    pass

def imageShowLoop():
    global cur_frame
    if type(cur_frame) != None:
        try:
            cv2.namedWindow("collect", 0)
            cv2.resizeWindow("collect", 1000, cur_frame.shape[0] * 1000 // cur_frame.shape[1])
            cv2.imshow("collect", cur_frame)
        except:
            pass
    main_window.after(50, imageShowLoop)

class MyCameraThread(threading.Thread):
    def __init__(self):
        super(MyCameraThread,self).__init__()

    def run(self):
        cap = cv2.VideoCapture(camera_id)
        ret, image = cap.read()
        if not ret:
            print("cannot read image")
            sys.exit()
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                print("cannot read image")
                continue
            global cur_frame
            lock.acquire()
            cur_frame = image
            lock.release()
            curtime = time.clock()
            global collect_state, video_out
            if collect_state == 2:
                if video_out != None:
                    video_out.release()
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    newfilename = str(curtime) + ".mp4"
                    video_out = cv2.VideoWriter(os.path.join(video_save_dir, newfilename), fourcc, 20, (image.shape[1], image.shape[0]))
                collect_state = 3
            elif collect_state == 3:
                if video_out != None:
                    video_out.write(image)
                else:
                    print("视频文件未创建")
            elif collect_state == 4:
                if video_out != None:
                    video_out.release()
                    video_out = None
            else:
                if video_out != None:
                    video_out.release()
                    video_out = None
            time.sleep(0.05)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                cv2.destroyAllWindows()
                break




frame_main = Frame(main_window,borderwidth=1,relief=SUNKEN)
frame_main.place(x = 0,y=0,width = window_width,height = window_height)

# open_camera_button = Button(frame_main,text ="打开相机", command = open_camera_click)
# open_camera_button.grid(row = 0 ,column=0)
#
# close_cameara_button = Button(frame_main,text ="关闭相机", command = close_camera_click)
# close_cameara_button.grid(row = 1 ,column=0)

begin_collect_button = Button(frame_main,text ="开始采集", command = begin_collect_click)
begin_collect_button.grid(row = 2 ,column=0)

end_collect_button = Button(frame_main,text ="结束采集", command = end_collect_click)
end_collect_button.grid(row = 3 ,column=0)

imageShowLoop()

t = MyCameraThread()
t.start()

main_window.mainloop()