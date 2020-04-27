#-*-coding:utf-8-*-



import dataset
import tensorflow as tf
import tensorflow.contrib as tf_contrib

import time
from datetime import timedelta
import math
import random
import numpy as np
# conda install --channel https://conda.anaconda.org/menpo opencv3
#Adding Seed so that random initialization is consistent
from numpy.random import seed
seed(10)
from tensorflow import set_random_seed
set_random_seed(20)

import image_util

batch_size = 32

#Prepare input data
classes = ['pos','neg']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size_width = 64
img_size_height = 64
num_channels = 3
train_path='data/image/ears/1.0'
model_name = "./model/resnet10_phone_model/phone.ckpt"

# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, img_size_width,img_size_height, classes, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))


weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)


def conv(x, channels, kernel=3, stride=1, padding='SAME', use_bias=False, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)

        return x

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

def relu(x):
    return tf.nn.relu(x)

def resblock(x_init, channels, is_training=True, use_bias=False, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)

        x_init = tf.identity(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')

        return x + x_init

def ResNet10(x):
    x = conv(x,channels=64,scope="conv")
    # x = batch_norm(x)
    # x = relu(x)
    # x = tf.nn.max_pool(value=x,
    #                    ksize=[1, 2, 2, 1],
    #                    strides=[1, 2, 2, 1],
    #                    padding='SAME')

    x = resblock(x,channels=64, is_training=True, downsample=False, scope='resblock0')
    x = resblock(x,channels=128, is_training=True, downsample=True, scope='resblock1')
    x = resblock(x,channels=256, is_training=True, downsample=True, scope='resblock2')
    x = resblock(x,channels=512, is_training=True, downsample=True, scope='resblock3')

    # global pooling
    x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)

    with tf.variable_scope("logit"):
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=2, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=True)
        x=tf.nn.dropout(x,keep_prob=0.7)
        return x



total_iterations = 0
def train(num_iteration):
    x = tf.placeholder(tf.float32, shape=[None,img_size_height, img_size_width,num_channels], name='x')
    fc = ResNet10(x)

    ## labels
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    # 预测值
    y_pred = tf.nn.softmax(fc,name='y_pred')
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    #损失函数-交叉熵
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc,labels=y_true)
    cost = tf.reduce_mean(cross_entropy)

    # 优化器,学习率
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    #正确率
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    session.run(tf.global_variables_initializer())


    saver = tf.train.Saver()

    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)


        # 增强
        n_batch = []
        for b  in x_batch:
            b = image_util.random_augmentation(b)
            b = b.astype(np.float32)
            b = np.multiply(b, 1.0 / 255.0)
            n_batch.append(b)
        x_batch = n_batch
        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}

        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}

        _,t_cost = session.run([optimizer,cost], feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))

            acc = session.run(accuracy, feed_dict=feed_dict_tr)
            val_acc = session.run(accuracy, feed_dict=feed_dict_val)
            msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%},Trainning Cost:{3:.3f} Validation Accuracy: {4:>6.1%},  Validation Loss: {5:.3f}"
            print(msg.format(epoch + 1,i, acc,t_cost, val_acc, val_loss))

            saver.save(session, model_name,global_step=i)


    total_iterations += num_iteration

train(num_iteration=1000000)



