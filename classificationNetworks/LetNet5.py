# encoding=utf-8
__author__ = "WSX"
# CNN的初始网络（LetNet5）用于分类
"""LetNet5 结构：  cov 5*5 6 > avgpooling > cov 5*5 6 > avgpooling > fc 120 fc 84"""
import tensorflow as tf
import numpy as np
from utils import weight_variable,bias_variable,conv2d,avg_pooling   #utils导入卷积  池化操作

class LetNet5(object):
    def __init__(self):
        pass

    def build(self, input, is_dropout = False):  #is_dropout 是否dropout
        # 卷积层1    cov 5*5 6
        W_conv1 = weight_variable([5, 5, tf.shape(input)[-1], 6])  #cov 5*5 6
        b_conv1 = bias_variable([6])
        h_conv1 = tf.nn.relu(conv2d(input, W_conv1) + b_conv1)
        h_pool1 = avg_pooling(h_conv1)
        # 卷积层2    cov 5*5 6
        W_conv2 = weight_variable([5, 5, 6, 6])
        b_conv2 = bias_variable([6])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = avg_pooling(h_conv2)
        #全连接1    120
        W_fc1 = weight_variable([7*7*6, 120])  #卷积后图像大小
        b_fc1 = bias_variable([120])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*6])  #需要将卷积后的拉伸为一列

        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        if is_dropout: h_fc1 = tf.nn.dropout(h_fc1, 0.5)
        # 全连接2  84
        W_fc2 = weight_variable([120, 84])  # 卷积后图像大小
        b_fc2 = bias_variable([84])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        if is_dropout: h_fc2 = tf.nn.dropout(h_fc2, 0.5)
        # output  10
        W_fc3 = weight_variable([84, 10])  # 卷积后图像大小
        b_fc3 = bias_variable([10])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        return h_fc3
