import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

from  image_util import *


num_channels = 3


smoke_graph = tf.Graph()
smoke_session = tf.Session(graph=smoke_graph)

phone_graph = tf.Graph()
phone_session = tf.Session(graph=phone_graph)

smoke_model_path = "./model/smoke_model/smoke.ckpt-99944"
phone_model_path = './model/phone_model/phone.ckpt-99922'

def loadSmokeGragh():
    ## Let us restore the saved model
    with smoke_session.as_default():
        with smoke_session.graph.as_default():
            # Step-1: Recreate the network graph. At this step only graph is created.
            saver = tf.train.import_meta_graph(smoke_model_path+".meta")
            # Step-2: Now let's load the weights saved using the restore method.
            saver.restore(smoke_session, smoke_model_path)
        # Accessing the default graph which we have restored

def loadPhoneGraph():
    ## Let us restore the saved model
    with phone_session.as_default():
        with phone_session.graph.as_default():
            # Step-1: Recreate the network graph. At this step only graph is created.
            saver = tf.train.import_meta_graph(phone_model_path + '.meta')
            # Step-2: Now let's load the weights saved using the restore method.
            saver.restore(phone_session, phone_model_path)

loadSmokeGragh()
loadPhoneGraph()


def detect_smoke(image):
    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    with smoke_session.as_default():
        with smoke_session.graph.as_default():
            graph = tf.get_default_graph()
            y_pred = graph.get_tensor_by_name("y_pred:0")

            ## Let's feed the images to the input placeholders
            x= graph.get_tensor_by_name("x:0")
            y_true = graph.get_tensor_by_name("y_true:0")
            y_test_images = np.zeros((1, 2))

            images = []
            # path = './smoke/neg/smoke_1300.png'
            # image = cv2.imread(path)
            # Resizing the image to our desired size and preprocessing will be done exactly as done during training
            image = cv2.resize(image, (mouth_width, mouth_height), 0, 0, cv2.INTER_LINEAR)
            images.append(image)
            images = np.array(images, dtype=np.uint8)
            images = images.astype('float32')
            images = np.multiply(images, 1.0 / 255.0)
            # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
            x_batch = images.reshape(1, mouth_height, mouth_width, num_channels)

            ### Creating the feed_dict that is required to be fed to calculate y_pred
            feed_dict_testing = {x: x_batch, y_true: y_test_images}
            result = smoke_session.run(y_pred, feed_dict=feed_dict_testing)
            # result is of this format [probabiliy_of_rose probability_of_sunflower]
            # dog [1 0]
            res_label = ['pos', 'neg']

            #print(res_label[result.argmax()])
            # if res_label[result.argmax()] == "pos":
            #     return True

            # if result.argmax() == 0:
            #     return True

            if result[0][0] >0.6:
                return True
            return False




def detect_phone(image):
    with phone_session.as_default():
        with phone_session.graph.as_default():
            graph = tf.get_default_graph()
            # Now, let's get hold of the op that we can be processed to get the output.
            # In the original network y_pred is the tensor that is the prediction of the network
            y_pred = graph.get_tensor_by_name("y_pred:0")

            ## Let's feed the images to the input placeholders
            x= graph.get_tensor_by_name("x:0")
            y_true = graph.get_tensor_by_name("y_true:0")
            y_test_images = np.zeros((1, 2))

            images = []
            # path = './smoke/neg/smoke_1300.png'
            # image = cv2.imread(path)
            # Resizing the image to our desired size and preprocessing will be done exactly as done during training
            image = cv2.resize(image, (ear_width, ear_height), 0, 0, cv2.INTER_LINEAR)
            images.append(image)
            images = np.array(images, dtype=np.uint8)
            images = images.astype('float32')
            images = np.multiply(images, 1.0 / 255.0)
            # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
            x_batch = images.reshape(1, ear_height, ear_width, num_channels)

            ### Creating the feed_dict that is required to be fed to calculate y_pred
            feed_dict_testing = {x: x_batch, y_true: y_test_images}
            result = phone_session.run(y_pred, feed_dict=feed_dict_testing)
            # result is of this format [probabiliy_of_rose probability_of_sunflower]
            # dog [1 0]
            res_label = ['pos', 'neg']

            #print(res_label[result.argmax()])
            # if res_label[result.argmax()] == "pos":
            #     return True
            # if result[0][0] >0.2:
            #     print(result)
            # if result.argmax() == 0:
            #     return True

            if result[0][0] > 0.6:
                return True
            return False