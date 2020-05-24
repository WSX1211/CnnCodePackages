# encoding=utf-8
__author__ = "WSX"
import tensorflow as tf
import numpy as np
from utils import max_pooling, fc ,conv3_3  #utils导入卷积  池化操作

class resnet(object):
    #输出类别个数 以及  是否需要 finetune
    def __init__(self,n_classes,batchsize, finetune = False):
        self.n_classes = n_classes
        self.finetune = finetune
        self.batchsize = batchsize
        try:
            self.data_dict_resnet = np.load('resnet.npy', allow_pickle=True, encoding='latin1').item()
            print("finetune load success...")
        except Exception as e:
            self.finetune = False

    def build(self, input, is_dropout = False):  #is_dropout 是否dropout
