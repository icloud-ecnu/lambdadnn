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
from ResNet import ResNet50
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

def receive_data(childPipe):
    context = zmq.Context()
    sub_recv = context.socket(zmq.SUB)
    sub_recv.connect("tcp://" + IP + ":5557")
    sub_recv.setsockopt(zmq.SUBSCRIBE, b'')
    while True:
        b_data_array = sub_recv.recv_pyobj()
        childPipe.send(b_data_array)
        continue

global model
model = ResNet50(input_shape = (32, 32, 3),classes = 10)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
def get_gradient_func(model):
    grads = K.gradients(model.total_loss, model.trainable_weights)
    inputs = model._feed_inputs + model._feed_targets + model._feed_sample_weights
    func = K.function(inputs, grads)
    return func
def trainHandler(event,context):
    worker_ids  = event["worker_ids"]  # the allocate work id
    bat_size = event["batch_size"]
    snumber = event["sub_data_size"]
    iteration_num = event["iteration_num"]
    itr_epo_mini = snumber//bat_size    #iterations of one epoch in the function training data

    parentPipe, childPipe = Pipe(False)

    time_start = time.time()
    boto3.Session().resource('s3').Bucket(BUCKET).Object(
        s3_data_path).download_file(local_data_path)

    boto3.Session().resource('s3').Bucket(BUCKET).Object(
        s3_lable_path).download_file(local_lable_path)

    train_images = np.load(local_data_path, allow_pickle=True)
    train_labels = np.load(local_lable_path, allow_pickle=True)
    train_images= train_images / 255.0

    context = zmq.Context()
    # send work
    sub_sender = context.socket(zmq.PUSH)
    sub_sender.connect("tcp://"+IP+":5558")

    socket_receive = multiprocessing.Process(name='socket_receive', target=receive_data,args=(childPipe,))
    socket_receive.start()
    init_m = model.get_weights()
    init_m = np.array(init_m)
    sub_sender.send_pyobj(init_m,protocol=4)
    print("worker id:",worker_ids )
    for itr_num_ini in range(1,iteration_num+1):
        itr_num = itr_num_ini % itr_epo_mini
        if itr_num_ini % itr_epo_mini == 0:
            itr_num = itr_epo_mini
        print("iteration number=", itr_num_ini)
        data = parentPipe.recv()
        data = np.array(data)
        weight_old = data
        if itr_num_ini == 1:
            m_weight_old = weight_old
        else:
            learningrate = model.optimizer.get_config()["lr"]
            m_weight_old = np.array(model.get_weights()) + learningrate*weight_old
        model.set_weights(m_weight_old)
        model.fit(train_images[(itr_num - 1) * bat_size: (itr_num) * bat_size, :, :, :],
                            train_labels[(itr_num - 1) * bat_size: (itr_num) * bat_size], batch_size=bat_size, epochs=1,
                            verbose=0)
        x = train_images[(itr_num - 1) * bat_size: (itr_num) * bat_size, :, :, :]
        y = train_labels[(itr_num - 1) * bat_size: (itr_num) * bat_size]
        get_gradient = get_gradient_func(model)
        gradient_dis = get_gradient([x, y, np.ones(len(y))])
        m_weight_new = np.array(gradient_dis)
        sub_sender.send_pyobj(m_weight_new, protocol=4)
    print("total finish time is ", round(time.time() - time_start, 4))
    socket_receive.terminate()
    return True

