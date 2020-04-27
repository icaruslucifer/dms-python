### 概述
一款开源的疲劳驾驶检测算法，包含训练、推理、android应用全套代码。


### 依赖
- openvino 2019 R1
- tensorflow 1.8.0及以上
- python opencv
- python  zmq


### 文件结构
train.py 训练检测抽烟和打电话的模型
predict.py 预测抽烟和打电话模型
detect_video.py dms主程序,用来测试使用
train_resnet10.py 用resnet10 模型来训练
area_util.py 根据landmarks计算眼部,耳部,嘴部的检测区域
collect_video.py 采集视频
rename.py 重命名文件
dataset.py 处理数据集
icarus_point.py 点类,线类
image_augmentation.py 数据增强类
image_util.py 图像处理类
label_image.py 从视频中取出图片进行正负样本标签,取出的是整张图片
label_video.py 从视频中取出图片进行正负样本标签,取出的是具体样本,后面很少使用,常用label_image
openvino_util.py 加载openvino的深度学习模型
product.py 产品级文件,取单个人脸,告警通过累计帧获取,告警发送zmq消息,同时通过zmq发送原始图像
zmq_image.py 通过zmq发送图像
zmq_test.py 测试zmq
android:配合android进行dms报警



