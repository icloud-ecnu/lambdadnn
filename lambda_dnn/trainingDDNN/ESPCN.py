# -*- coding: utf-8 -*-
import os
import json
import time
import functools
import tarfile
import sys
import subprocess
import multiprocessing
import threading
from multiprocessing import Process, Pipe
import boto3
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import socket
import struct
import zmq


import sys
from six.moves import cPickle
from tensorflow.keras import backend as K
from tensorflow.keras import losses


BUCKET = os.environ['BUCKET']
IP = os.environ['IP']
FILE_DIR = '/tmp/'
PSPATH= '/home/ec2-user/upd/'
s3 = boto3.resource('s3')
s3_data_path=""
s3_lable_path=""
local_data_path = ""
local_lable_path =""

crop_size = 300
upscale_factor = 3
input_size = crop_size // upscale_factor


def receive_data(childPipe):
    context = zmq.Context()
    sub_recv = context.socket(zmq.SUB)
    sub_recv.connect("tcp://" + IP + ":5557")
    sub_recv.setsockopt(zmq.SUBSCRIBE, b'')
    while True:
        b_data_array = sub_recv.recv_pyobj()
        childPipe.send(b_data_array)
        continue

# Use TF Ops to process.
def process_input(input, input_size, upscale_factor):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize_images(y, [input_size, input_size], method=tf.image.ResizeMethod.AREA)

def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y

def get_model(upscale_factor, channels):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = tf.keras.Input(shape=(None,None,1))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    Subpixel_layer = tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,upscale_factor))
    outputs = Subpixel_layer(inputs=x)

    return tf.keras.Model(inputs, outputs)


global model
model = get_model(upscale_factor=upscale_factor, channels=1)
loss_fn = losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer, loss=loss_fn)

def get_gradient_func(model):
    grads = K.gradients(model.total_loss, model.trainable_weights)
    inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
    func = K.function(inputs, grads)
    return func
def trainHandler(event,context):
    worker_ids  = event["worker_ids"]
    bat_size = event["batch_size"]
    snumber = event["sub_data_size"]
    iteration_num = event["iteration_num"]
    itr_epo_mini = snumber//bat_size    #iterations of one epoch in the function training data

    parentPipe, childPipe = Pipe(False)
    context = zmq.Context()
    # send
    sub_sender = context.socket(zmq.PUSH)
    sub_sender.connect("tcp://"+IP+":5558")

    socket_receive = multiprocessing.Process(name='socket_receive', target=receive_data,args=(childPipe,))
    socket_receive.start()

    time_start = time.time()
    boto3.Session().resource('s3').Bucket(BUCKET).Object(
        s3_data_path).download_file(local_data_path)

    boto3.Session().resource('s3').Bucket(BUCKET).Object(
        s3_lable_path).download_file(local_lable_path)

    boto3.Session().resource('s3').Bucket(BUCKET).Object(
        s3_lable_path).download_file(
        local_lable_path)

    train_ds = np.load(local_lable_path, allow_pickle=True)
    train_ds = tf.constant(train_ds, dtype=tf.float32)
    train_ds = train_ds / 255.0
    train_ds = tf.convert_to_tensor(train_ds, dtype=tf.float32)
    train_images = process_input(train_ds, input_size, upscale_factor)
    train_labels = process_target(train_ds)

    init_m = model.get_weights()
    init_m = np.array(init_m)
    sub_sender.send_pyobj(init_m,protocol=4)
    print("worker id:",worker_ids )
    for itr_num_ini in range(1,iteration_num+1):
        itr_num = itr_num_ini % itr_epo_mini
        if itr_num_ini % itr_epo_mini == 0:
            itr_num = itr_epo_mini
        #print("iteration number=", itr_num_ini)
        data = parentPipe.recv()
        data = np.array(data)
        weight_old = data
        if itr_num_ini == 1:
            m_weight_old = weight_old
        else:
            learningrate = model.optimizer.get_config()["lr"]
            m_weight_old = np.array(
                model.get_weights()) + learningrate * weight_old
        model.set_weights(m_weight_old)
        model.fit(train_images[(itr_num - 1) * bat_size: (itr_num) * bat_size],
                  train_labels[(itr_num - 1) * bat_size: (itr_num) * bat_size], batch_size=bat_size, epochs=1,
                  verbose=0, steps_per_epoch=itr_epo_mini)
        x = train_images[(itr_num - 1) * bat_size: (itr_num) * bat_size]
        y = train_labels[(itr_num - 1) * bat_size: (itr_num) * bat_size]
        get_gradient = get_gradient_func(model)
        gradient_dis = get_gradient([x, y, np.ones(len(y))])
        m_weight_new = np.array(gradient_dis)
        sub_sender.send_pyobj(m_weight_new, protocol=4)
    print("total finish time is ", round(time.time() - time_start, 4))
    socket_receive.terminate()
    return True




