# encoding=utf-8
__author__ = "WSX"

import os
import tensorflow as tf
from PIL import Image
import sys


def creat_tf(imgpath):
    cwd = os.getcwd()  #获取当前路径
    classes = os.listdir(cwd + imgpath)

    # 此处定义tfrecords文件存放
    writer = tf.python_io.TFRecordWriter("./" + imgpath + "/" +"train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd + imgpath + name + "/"
        print(class_path)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize((224, 224))
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
                writer.write(example.SerializeToString())
                print(img_name)
    writer.close()


def read_example():
    # 简单的读取例子：
    for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        image = example.features.feature['img_raw'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        # 可以做一些预处理之类的


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    # 转换为float32类型，并做归一化处理
    img = tf.cast(img, tf.float32)  # * (1. / 255)
    label = tf.cast(features['label'], tf.int64)
    return img, label


def menu():
    while True:
        print("请选择操作：  1 制作数据集   2 读取数据集   3 退出")
        raw = input()
        if raw == "1":
            creat_tf("datasets/dataset1")
            print("creat success!")
        elif raw == "2":
            print("read success!")
        elif raw == "3":
            break
        else:
            print("ERROR！ please try again")