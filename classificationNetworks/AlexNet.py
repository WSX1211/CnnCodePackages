# encoding=utf-8
__author__ = "WSX"
"""AlexNet 结构：  
输入层：227×227×3
C1：96×11×11×3（卷积核个数/宽/高/厚度） relu LRN pool     卷积步长s_c= 4    池化步长s_p=  2
C2：256×5×5×96（卷积核个数/宽/高/厚度） relu LRN pool   
C3：384×3×3×256（卷积核个数/宽/高/厚度）relu
C4：384×3×3×384（卷积核个数/宽/高/厚度）relu
C5：256×3×3×384（卷积核个数/宽/高/厚度）relu - pool
fc1：4096  relu - dropout
fc2：4096  relu - dropout
所有池化为 3*3 大小  s_p = 2
"""
import tensorflow as tf
import numpy as np
from utils import max_pooling, fc   #utils导入卷积  池化操作
from utils import convM_N, conv3_3

class AlexNet(object):
    # 输出类别个数 以及  是否需要 finetune  batchsize作用是为了卷积转化全连接时无需计算维度
    def __init__(self, n_classes, batchsize, finetune = False):
        self.finetune = finetune
        self.n_classes = n_classes
        self.batchsize = batchsize
        try:
            self.data_dict_AlexNet = np.load('AlexNet.npy', allow_pickle=True, encoding='latin1').item()
            print("finetune load success...")
        except Exception as e:
            self.finetune = False

    def build(self, input, is_dropout=False):  # is_dropout 是否dropout
        #卷积层1
        conv1 = convM_N(input, 96, "conv1", self.data_dict_AlexNet, [11,11], 4, finetune = self.finetune)
        lrn1 = tf.nn.lrn(conv1, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, [1,3,3,1], [1,2,2,1],padding='VALID',name = 'pool1')
        # 卷积层2
        conv2 = convM_N(pool1, 256, "conv2", self.data_dict_AlexNet, [5,5], 1, finetune = self.finetune)
        lrn2 = tf.nn.lrn(conv2, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
        pool2 = tf.nn.max_pool(lrn2, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool2')
        #卷积层3
        conv3 = conv3_3(pool2, 384, 'conv3', self.data_dict_AlexNet, finetune=self.finetune)
        #卷积层4
        conv4 = conv3_3(conv3, 384, 'conv4', self.data_dict_AlexNet, finetune=self.finetune)
        #卷积层5
        conv5 = conv3_3(conv4, 256, 'conv5', self.data_dict_AlexNet, finetune=self.finetune)
        pool3 = tf.nn.max_pool(conv5, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool3')

        # fully connected layer 全连接
        flatten = tf.reshape(pool3, [self.batchsize, -1])
        fc1 = fc(flatten, 4096, 'fc1', finetune=False)
        fc1 = tf.nn.relu(fc1)
        if is_dropout: fc1 = tf.nn.dropout(fc1, 0.5)

        fc2 = fc(fc1, 4096, 'fc2', finetune=False)
        fc2 = tf.nn.relu(fc2)
        if is_dropout: fc2 = tf.nn.dropout(fc2, 0.5)

        fc3 = fc(fc2, self.n_classes, 'fc3', finetune=False)
        return fc3


