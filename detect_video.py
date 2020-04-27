# _*_ coding=UTF-8 _*_


import area_util

import predict
import openvino_util
from  image_util import *



label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
landmark_color = (0, 0, 255)
box_thickness = 1


need_collect = False
video_out = None
video_save_dir = "./videos"
if not os.path.exists(video_save_dir):
    os.mkdir(video_save_dir)



def isYield(pos):
    if abs(pos[1]) > 30:
        return True
    return False

def isGlance(pos):
    if abs(pos[2]) > 20:
        return True
    return False

# 是否打哈欠
def isYawn(landmark):
    w = ((landmark[8][0]-landmark[9][0])**2 + (landmark[8][1]-landmark[9][1])**2)**0.5
    h = ((landmark[10][0]-landmark[11][0])**2 + (landmark[10][1]-landmark[11][1])**2)**0.5

    if w/h > 1.8:
        return False
    return True


def main():
    cap = cv2.VideoCapture(0)
    ret, image = cap.read()
    if not ret:
        print ("cannot read image")
        sys.exit()

    if float(image.shape[1])/image.shape[0] < float(openvino_util.face_input_width)/openvino_util.face_input_height:
        new_w = image.shape[1]*openvino_util.face_input_height//image.shape[0]
        new_h = openvino_util.face_input_height
    else:
        new_w = openvino_util.face_input_width
        new_h = image.shape[0] * openvino_util.face_input_width // image.shape[1]

    face_input_v ,face_exec_net,face_plugin = openvino_util.loadfacemodel("CPU")
    landmark_input_v,landmark_exec_net,landmark_plugin = openvino_util.loadfacelandmarkmodel("CPU")
    headpose_input_v,headpose_exec_net,headpose_plugin = openvino_util.loadHeadPoseModel("CPU")

    while cap.isOpened():
        ret, oriimage = cap.read()
        if not ret:
            print ("cannot read image")
            break
        try:
            image = oriimage.copy()
            resized_image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
            canvas = np.full((openvino_util.face_input_height, openvino_util.face_input_width, 3), 128)
            canvas[(openvino_util.face_input_height-new_h)//2:(openvino_util.face_input_height-new_h)//2 + new_h,(openvino_util.face_input_width-new_w)//2:(openvino_util.face_input_width-new_w)//2 + new_w,  :] = resized_image
            prepimg = canvas
            prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
            prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
            start =time.time()
            outputs = face_exec_net.infer(inputs={face_input_v: prepimg})
            #print('face detect time:'+str(time.time()-start))
            faces = openvino_util.parseFaceOutput(outputs,new_h,new_w,image.shape[0],image.shape[1],0.7)

            for face in faces:
                face_image = image[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
                cv2.rectangle(image, (face[0], face[1]), (face[0]+face[2], face[1]+face[3]), box_color, box_thickness)
                # 检测人脸
                f_size = face_image.shape[0]
                if face_image.shape[0]<face_image.shape[1]:
                    f_size = face_image.shape[1]
                canvas = np.full((f_size, f_size, 3), 128).astype(np.uint8)
                canvas[(f_size-face_image.shape[0])//2:(f_size-face_image.shape[0])//2 + face_image.shape[0],(f_size-face_image.shape[1])//2:(f_size-face_image.shape[1])//2 + face_image.shape[1],  :] = face_image
                resized_image = cv2.resize(canvas, (openvino_util.landmark_input_size, openvino_util.landmark_input_size), interpolation = cv2.INTER_CUBIC)
                prepimg = resized_image
                prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
                prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
                start =time.time()
                outputs = landmark_exec_net.infer(inputs={landmark_input_v: prepimg})
                #print('landmark detect time:'+str(time.time()-start))
                landmarks = openvino_util.parseLandmarkOutput(outputs,face_image.shape[0],face_image.shape[1])
                t_landmarks = []
                for landmark in landmarks:
                    x = landmark[0]+ face[0]
                    y = landmark[1]+ face[1]
                    t_landmarks.append([x,y])
                    cv2.circle(image,(x,y),2,landmark_color, box_thickness)

                if isYawn(t_landmarks):
                    cv2.putText(image, "yawn", (image.shape[1]-120, 150), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                # 检测姿态
                start =time.time()
                headpose_outputs = headpose_exec_net.infer(inputs={headpose_input_v: prepimg})
                rol,pitch,yaw = openvino_util.parseHeadPoseOutput(headpose_outputs)
                #print('headpose detect time:'+str(time.time()-start))

                cv2.putText(image, str(int(rol)), (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, str(int(pitch)), (30, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, str(int(yaw)), (30, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                if isYield([rol,pitch,yaw]):
                    cv2.putText(image, "yield", (image.shape[1]-120, 180), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                if isGlance([rol,pitch,yaw]):
                    cv2.putText(image, "glance", (image.shape[1]-120, 210), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

                # 检测眼睛
                eyes_detectarea = area_util.getEyesArea(t_landmarks)
                cv2.line(image, (eyes_detectarea[0], eyes_detectarea[1]), (eyes_detectarea[2], eyes_detectarea[3]), box_color, box_thickness)
                cv2.line(image, (eyes_detectarea[2], eyes_detectarea[3]), (eyes_detectarea[4], eyes_detectarea[5]), box_color, box_thickness)
                cv2.line(image, (eyes_detectarea[4], eyes_detectarea[5]), (eyes_detectarea[6], eyes_detectarea[7]), box_color, box_thickness)
                cv2.line(image, (eyes_detectarea[6], eyes_detectarea[7]), (eyes_detectarea[0], eyes_detectarea[1]), box_color, box_thickness)

                # 检查打电话
                leftphone_detectarea = area_util.getLeftPhoneDetectArea(t_landmarks)
                cv2.line(image, (leftphone_detectarea[0], leftphone_detectarea[1]), (leftphone_detectarea[2], leftphone_detectarea[3]), box_color, box_thickness)
                cv2.line(image, (leftphone_detectarea[2], leftphone_detectarea[3]), (leftphone_detectarea[4], leftphone_detectarea[5]), box_color, box_thickness)
                cv2.line(image, (leftphone_detectarea[4], leftphone_detectarea[5]), (leftphone_detectarea[6], leftphone_detectarea[7]), box_color, box_thickness)
                cv2.line(image, (leftphone_detectarea[6], leftphone_detectarea[7]), (leftphone_detectarea[0], leftphone_detectarea[1]), box_color, box_thickness)

                leftphonecall_img = area_util.getLeftPhoneImage(oriimage,t_landmarks)

                if leftphonecall_img.shape[0] != 0 and leftphonecall_img.shape[1] !=0:
                    phone_img1 = cv2.resize(leftphonecall_img, (ear_width, ear_height))
                    start =time.time()
                    isphone1 = predict.detect_phone(phone_img1)
                    #print('phone detect time:'+str(time.time()-start))
                else:
                    isphone1 = False

                rightphone_detectarea = area_util.getRightPhoneDetectArea(t_landmarks)
                cv2.line(image, (rightphone_detectarea[0], rightphone_detectarea[1]), (rightphone_detectarea[2], rightphone_detectarea[3]), box_color, box_thickness)
                cv2.line(image, (rightphone_detectarea[2], rightphone_detectarea[3]), (rightphone_detectarea[4], rightphone_detectarea[5]), box_color, box_thickness)
                cv2.line(image, (rightphone_detectarea[4], rightphone_detectarea[5]), (rightphone_detectarea[6], rightphone_detectarea[7]), box_color, box_thickness)
                cv2.line(image, (rightphone_detectarea[6], rightphone_detectarea[7]), (rightphone_detectarea[0], rightphone_detectarea[1]), box_color, box_thickness)

                rightphonecall_img = area_util.getRightPhoneImage(oriimage,t_landmarks)
                # try:
                #     cv2.imshow("left_phone", rightphonecall_img)
                # except:
                #     pass

                if rightphonecall_img.shape[0] != 0 and rightphonecall_img.shape[1] !=0:
                    phone_img2 = cv2.resize(rightphonecall_img, (ear_width, ear_height))
                    start =time.time()
                    isphone2 = predict.detect_phone(phone_img2)
                    #print('phone detect time:'+str(time.time()-start))
                else:
                    isphone2 = False
                if isphone1 or isphone2:
                    if isphone1:
                        cv2.putText(image, "leftphone", (image.shape[1]-120, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                    if isphone2:
                        cv2.putText(image, "rightphone", (image.shape[1]-120, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)


                # 检测抽烟
                smokedetectArea = area_util.getSmokeDetectArea(t_landmarks)
                cv2.line(image, (smokedetectArea[0], smokedetectArea[1]), (smokedetectArea[2], smokedetectArea[3]), (255,255,0), box_thickness)
                cv2.line(image, (smokedetectArea[2], smokedetectArea[3]), (smokedetectArea[4], smokedetectArea[5]), (255,255,0), box_thickness)
                cv2.line(image, (smokedetectArea[4], smokedetectArea[5]), (smokedetectArea[6], smokedetectArea[7]), (255,255,0), box_thickness)
                cv2.line(image, (smokedetectArea[6], smokedetectArea[7]), (smokedetectArea[0], smokedetectArea[1]), (255,255,0), box_thickness)

                smoke_image = area_util.getSmokeImage(oriimage,t_landmarks)
                # if smoke_image.shape[0] > 0:
                #     try:
                #         cv2.imshow("mouth", smoke_image)
                #     except:
                #         pass
                #cv2.imwrite(str(time.clock())+".png", smoke_image)
                if smoke_image.shape[0] != 0 and smoke_image.shape[1] !=0:
                    smoke_image = cv2.resize(smoke_image, (mouth_width, mouth_height))
                    start =time.time()
                    issmoke = predict.detect_smoke(smoke_image)
                    #print('smoke detect time:'+str(time.time()-start))
                else:
                    issmoke = False
                if issmoke :
                    cv2.putText(image, "smoke", (image.shape[1]-120, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
        except:
            continue
        cv2.namedWindow("Result", 0)
        cv2.resizeWindow("Result", 1000, image.shape[0]*1000//image.shape[1])
        cv2.imshow("Result", image)

        global need_collect
        k = cv2.waitKey(1)&0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            need_collect = True
        elif k == ord('t'):
            need_collect = False

        global video_out
        if need_collect:
            if video_out == None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                newfilename = str(time.time()) + ".mp4"
                video_out = cv2.VideoWriter(os.path.join(video_save_dir, newfilename), fourcc, 20, (image.shape[1], image.shape[0]))
            video_out.write(oriimage)
        else:
            if video_out != None:
                video_out.release()
            video_out = None
    cv2.destroyAllWindows()
    del headpose_exec_net,face_exec_net,landmark_exec_net
    del headpose_plugin,face_plugin,landmark_plugin


if __name__ == '__main__':
    main()