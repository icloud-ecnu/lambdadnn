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
from tensorflow.keras.applications.mobilenet import MobileNet
import socket
import struct
import zmq
import math

import sys
from six.moves import cPickle
from tensorflow.keras import backend as K
from tensorflow.keras import losses

time0 = time.time()
BUCKET = os.environ['BUCKET']
IP = os.environ['IP']
FILE_DIR = '/tmp/'
s3 = boto3.resource('s3')
s3_data_path=""
s3_lable_path=""
local_data_path = ""
local_lable_path =""
check_point_model_path=""
local_model_path = ""

def receive_data(childPipe):
    context = zmq.Context()
    sub_recv = context.socket(zmq.SUB)
    sub_recv.connect("tcp://" + IP + ":5557")
    sub_recv.setsockopt(zmq.SUBSCRIBE, b'')
    while True:
        b_data_array = sub_recv.recv_pyobj()
        childPipe.send(b_data_array)
        #print("send to pipe")
        continue

global model
model = MobileNet(weights=None,input_shape = (32, 32, 3),classes = 10)
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
    max_fnum = event["fuction_number"]
    snumber = event["sub_data_size"]
    iteration_num = event["iteration_num"]
    next_iter_num = event["next_iter_num"]  # first is 1,
    index = event["index"]
    delta_time_total = event["delta_time_total"]
    delta_time = event["delta_time"]
    itr_epo_mini = snumber//bat_size    #iterations of one epoch in the function training data

    parentPipe, childPipe = Pipe(False)
    context = zmq.Context()
    # send work
    sub_sender = context.socket(zmq.PUSH)
    sub_sender.connect("tcp://" + IP + ":5558")
    socket_receive = multiprocessing.Process(name='socket_receive', target=receive_data, args=(childPipe,))
    socket_receive.start()
    if next_iter_num == 1:
        init_m = model.get_weights()
        init_m = np.array(init_m)
        sub_sender.send_pyobj(init_m, protocol=4)
    else:
        boto3.Session().resource('s3').Bucket(BUCKET).Object(
            check_point_model_path).download_file(local_model_path)
        init_m = np.load(local_model_path,allow_pickle=True)
        model.set_weights(init_m)
        lastweight = np.array(model.get_weights())
        sub_sender.send_pyobj(lastweight, protocol=4)

    time_start = time.time()
    boto3.Session().resource('s3').Bucket(BUCKET).Object(
        s3_data_path).download_file(local_data_path)

    boto3.Session().resource('s3').Bucket(BUCKET).Object(
        s3_lable_path).download_file(local_lable_path)

    train_images = np.load(local_data_path, allow_pickle=True)
    train_labels = np.load(local_lable_path, allow_pickle=True)
    train_images= train_images / 255.0
    init_m = model.get_weights()
    init_m = np.array(init_m)
    sub_sender.send_pyobj(init_m,protocol=4)

    for itr_num_ini in range(next_iter_num,iteration_num+1):
        time1 = time.time()
        timebound = 860 * index
        if time1 - time0 >= timebound and (iteration_num - itr_num_ini + 1) * delta_time >= 50:
            weight_last = np.array(model.get_weights())  # model weight, not delta weight
            np.save(local_model_path, weight_last)
            boto3.Session().resource('s3').Bucket(BUCKET).Object(check_point_model_path).upload_file(local_model_path)
            split_number = max_fnum
            index += 1
            payload_w = {
                "filedir": "cifar_data_" + str(split_number),
                "epoch_num": 1,
                "worker_ids": worker_ids,
                "batch_size": bat_size,
                "fuction_number": max_fnum,
                "sub_data_size": snumber,
                "iteration_num": iteration_num,
                "next_iter_num": itr_num_ini,
                "index": index,
                "delta_time_total": delta_time_total,
                "delta_time": delta_time
            }
            boto3.client('lambda').invoke(FunctionName='functionname:$LATEST', InvocationType='Event',
                                          Payload=json.dumps(payload_w))
            index += 1
            break
        #print("worker_id: %d, iteration_number: %d" % (worker_ids, itr_num_ini))  # print make duration long
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
        time2 = time.time()
        delta_time_total += round(time2 - time1, 2)
        delta_time = math.ceil(delta_time_total / (
                itr_num_ini - next_iter_num + 1))  # average history iteration time as predict next iteration time
        x = train_images[(itr_num - 1) * bat_size: (itr_num) * bat_size, :, :, :]
        y = train_labels[(itr_num - 1) * bat_size: (itr_num) * bat_size]
        get_gradient = get_gradient_func(model)
        gradient_dis = get_gradient([x, y, np.ones(len(y))])
        m_weight_new = np.array(gradient_dis)
        sub_sender.send_pyobj(m_weight_new, protocol=4)
    print("total finish time is ", round(time.time() - time_start, 4))
    socket_receive.terminate()
    return True
