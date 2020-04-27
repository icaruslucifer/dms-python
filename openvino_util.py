# _*_ coding=UTF-8 _*_

import sys,os,cv2,time
import  numpy as np
from openvino.inference_engine import IENetwork,IEPlugin
import platform

landmark_input_size = 60

face_input_width = 672
face_input_height = 384


def loadHeadPoseModel(device):
    plugin = IEPlugin(device=device)
    if "CPU" in device:
        if 'Darwin' in  platform.platform():
            plugin.add_cpu_extension("/opt/intel/openvino_2019.1.144/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib")
        else:
            plugin.add_cpu_extension("lib/libcpu_extension.so")
        model_xml = "model/face_model/R1/head_pose/FP32/head-pose-estimation-adas-0001.xml" #<--- CPU
    else:
        model_xml = "model/face_model/R1/head_pose/FP16/head-pose-estimation-adas-0001.xml" #<--- MYRIAD
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)
    return input_blob,exec_net,plugin

def loadfacelandmarkmodel(device):
    plugin = IEPlugin(device=device)
    if "CPU" in device:
        if 'Darwin' in  platform.platform():
            plugin.add_cpu_extension("/opt/intel/openvino_2019.1.144/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib")
        else:
            plugin.add_cpu_extension("lib/libcpu_extension.so")
        model_xml = "model/face_model/R1/face_landmark_35/FP32/facial-landmarks-35-adas-0002.xml" #<--- CPU
    else:
        model_xml = "model/face_model/R1/face_landmark_35/FP16/facial-landmarks-35-adas-0002.xml" #<--- MYRIAD
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)
    return input_blob,exec_net,plugin


def loadfacemodel(device):
    plugin = IEPlugin(device=device)
    if "CPU" in device:
        if 'Darwin' in  platform.platform():
            plugin.add_cpu_extension("/opt/intel/openvino_2019.1.144/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib")
        else:
            plugin.add_cpu_extension("lib/libcpu_extension.so")
        model_xml = "./model/face_model/R1/face/FP32/face-detection-adas-0001.xml" #<--- CPU
    else:
        model_xml = "./model/face_model/R1/face/FP16/face-detection-adas-0001.xml" #<--- MYRIAD
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)
    return input_blob,exec_net,plugin


def parseFaceOutput(output,resized_im_h, resized_im_w, original_im_h, original_im_w, threshold):
    detectors =output['detection_out']
    blobs_size = detectors.shape[2]
    faces = []
    for i in range(blobs_size):
        if detectors[0][0][i][2] > threshold:
            #print(detectors[0][0][i])
            # 比例值
            x = detectors[0][0][i][3]
            y = detectors[0][0][i][4]
            w = detectors[0][0][i][5]-x
            h = detectors[0][0][i][6]-y
            # 像素值
            x *=face_input_width
            y *=face_input_height
            w *=face_input_width
            h *= face_input_height
            # resize后的值
            x -= (face_input_width - resized_im_w)/2
            y -= (face_input_height - resized_im_h)/2
            # 比例缩放
            scale = original_im_w/resized_im_w
            x = x* scale
            y = y*scale
            w = w*scale
            h = h*scale
            faces.append([int(x),int(y),int(w),int(h)])
    return faces


def parseLandmarkOutput(outputs,ori_h,ori_w):
    landmarks = []
    res = outputs['align_fc3']
    for i in range(0,res.shape[1],2):
        x = res[0][i]
        y = res[0][i+1]
        x *=landmark_input_size
        y *=landmark_input_size

        ori_size = ori_w
        if ori_h>ori_w:
            ori_size = ori_h

        scale = ori_size/landmark_input_size

        x *= scale
        y *= scale

        x -= (ori_size-ori_w)/2
        y -= (ori_size-ori_h)/2
        landmarks.append([int(x),int(y)])
    return landmarks

def parseHeadPoseOutput(outputs):
    pitch = outputs['angle_p_fc'][0][0] # 头旋转
    yaw = outputs['angle_y_fc'][0][0]   # 头上下
    rol = outputs['angle_r_fc'][0][0]   # 头左右
    return rol,pitch,yaw