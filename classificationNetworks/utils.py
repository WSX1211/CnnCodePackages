# encoding=utf-8
__author__ = "WSX"

#存放一些辅助的函数
import tensorflow as tf
import numpy as np

# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积定义
def conv2d(x, W, name="conv2d"):  # x:输入  W:卷积核
    with tf.variable_scope(name):
        conv = tf.nn.conv2d(x, W, strides=[1,1,1,1],padding="SAME")
        return conv

# 池化定义
def max_pooling(x, name = "max_pooling"):
    with tf.variable_scope(name):
        max_pooling = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return max_pooling

def atrous_conv2d(x, W, rate = 2,name="conv2d"):  # x:输入  W:卷积核
    with tf.variable_scope(name):
        conv = tf.nn.atrous_conv2d(x, W, rate=rate,padding="SAME")
        return conv

def avg_pooling(x, name = "avg_pooling"):
    with tf.variable_scope(name):
        avg_pooling = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return avg_pooling

#含有finetune功能的卷积操作
def conv3_3(x, out_channel, name, data_dict, finetune=False):
    in_channel = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if finetune:
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
        else:
            weight = tf.Variable(tf.truncated_normal([3, 3, in_channel, out_channel], stddev=0.1), name="weights")
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[out_channel]), trainable=True, name="bias")
        conv = tf.nn.conv2d(x, weight, [1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        return activation

def fc(x, out_channel, name, data_dict, finetune=False):
    in_channel = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if finetune:
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
        else:
            weight = tf.Variable(tf.truncated_normal([in_channel, out_channel], stddev=0.1), name="weights")
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[out_channel]), trainable=True, name="bias")
        net = tf.add(tf.matmul(x, weight), bias)
        return net

#自定义方式1 卷积M*N 步长为 cov_strid 的卷积函数
def convM_N(x, out_channel, name, data_dict,keranl_size, strids,finetune=False):
    in_channel = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if finetune:
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
        else:
            weight = tf.Variable(tf.truncated_normal([keranl_size[0], keranl_size[1], in_channel, out_channel], stddev=0.1), name="weights")
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[out_channel]), trainable=True, name="bias")
        conv = tf.nn.conv2d(x, weight, [1, strids, strids, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        return activation

#自定义方式2 卷积M*N 步长为 cov_strid 的卷积函数
#shape为卷积核的尺寸
def convMN(x, shape, name, data_dict,strids,finetune=False):
    with tf.name_scope(name) as scope:
        if finetune:
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
        else:
            weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name="weights")
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[shape[3]]), trainable=True, name="bias")
        conv = tf.nn.conv2d(x, weight, strids, padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        return activation
