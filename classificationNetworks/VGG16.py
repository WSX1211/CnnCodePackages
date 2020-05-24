# encoding=utf-8
__author__ = "WSX"
# CNN的初始网络（VGG16）用于分类
"""VGG16 结构：  
2*cov 3*3 64 > maxpooling > 2*cov 3*3 128 > maxpooling 3*cov 3*3 256 > maxpooling  
3*cov 3*3 512 > maxpooling   3*cov 3*3 512 > maxpooling  fc 4096  fc 4096 fc 1000
"""
import tensorflow as tf
import numpy as np
from utils import max_pooling, fc   #utils导入卷积  池化操作
from utils import conv3_3
#加载与训练模型

class VGG16(object):
    #输出类别个数 以及  是否需要 finetune
    def __init__(self,n_classes,batchsize, finetune = False):
        self.n_classes = n_classes
        self.finetune = finetune
        self.batchsize = batchsize
        try:
            self.data_dict_VGG16 = np.load('vgg16.npy', allow_pickle=True, encoding='latin1').item()
            print("finetune load success...")
        except Exception as e:
            self.finetune = False

    def build(self, input, is_dropout = False):  #is_dropout 是否dropout
        conv1_1 = conv3_3(input, 64, 'conv1_1',self.data_dict_VGG16, finetune=self.finetune)
        conv1_2 = conv3_3(conv1_1, 64, 'conv1_2',self.data_dict_VGG16,  finetune=self.finetune)
        pool1 = max_pooling(conv1_2, 'pool1')
        # conv2
        conv2_1 = conv3_3(pool1, 128, 'conv2_1',self.data_dict_VGG16,  finetune=self.finetune)
        conv2_2 = conv3_3(conv2_1, 128, 'conv2_2',self.data_dict_VGG16,  finetune=self.finetune)
        pool2 = max_pooling(conv2_2, 'pool2')
        # conv3
        conv3_1 = conv3_3(pool2, 256, 'conv3_1',self.data_dict_VGG16,  finetune=self.finetune)
        conv3_2 = conv3_3(conv3_1, 256, 'conv3_2',self.data_dict_VGG16,  finetune=self.finetune)
        conv3_3 = conv3_3(conv3_2, 256, 'conv3_3',self.data_dict_VGG16,  finetune=self.finetune)
        pool3 = max_pooling(conv3_3, 'pool3')
        # conv4
        conv4_1 = conv3_3(pool3, 512, 'conv4_1', self.data_dict_VGG16, finetune=self.finetune)
        conv4_2 = conv3_3(conv4_1, 512, 'conv4_2', self.data_dict_VGG16, finetune=self.finetune)
        conv4_3 = conv3_3(conv4_2, 512, 'conv4_3', self.data_dict_VGG16, finetune=self.finetune)
        pool4 = max_pooling(conv4_3, 'pool4')

        # conv5
        conv5_1 = conv3_3(pool4, 512, 'conv5_1', self.data_dict_VGG16, finetune=self.finetune)
        conv5_2 = conv3_3(conv5_1, 512, 'conv5_2', self.data_dict_VGG16, finetune=self.finetune)
        conv5_3 = conv3_3(conv5_2, 512, 'conv5_3', self.data_dict_VGG16, finetune=self.finetune)
        pool5 = max_pooling(conv5_3, 'pool5')

        # fully connected layer
        flatten = tf.reshape(pool5, [self.batchsize, -1])
        fc_6 = fc(flatten, 4096, 'fc_6', finetune=False)
        fc_6 = tf.nn.relu(fc_6)
        if is_dropout: fc_6 = tf.nn.dropout(fc_6, 0.5)

        fc_7 = fc(fc_6, 4096, 'fc_7', finetune=False)
        fc_7 = tf.nn.relu(fc_7)
        if is_dropout: fc_7 = tf.nn.dropout(fc_7, 0.5)

        fc_8 = fc(fc_7, self.n_classes, 'fc_8', finetune=False)
        return fc_8

