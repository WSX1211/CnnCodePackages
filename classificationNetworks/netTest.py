# encoding=utf-8
__author__ = "WSX"
#同来进行网络测试的代码
import tensorflow as tf
import os
from creatDataset import read_and_decode
MODEL_SAVE_PATH = "./model/letnet5"
MODEL_NAME = "LetNet5"
STEPS = 100
graph_path = "./dirs"  #保存graph的文件夹
max_acc = 0  # 最高测试准确率测试
dataset = "./datasets/dataset1/"
IMG_W, IMG_H, CHANNELS, outs = [32,32,3,10]  #图像大小 和 维度 类别个数

def test(net):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [-1, IMG_W, IMG_H, CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, outs])
        y = net.build(x)  # 网络输出
        # 读取数据
        images_test, labels_test = read_and_decode(dataset + 'test.tfrecords')
        img_batch_test, label_batch_test = tf.train.shuffle_batch([images_test, labels_test],
                                                                  batch_size=1000,
                                                                  capacity=392,
                                                                  min_after_dequeue=200)
        # 使用滑动平均模型  进行模型读取和保存的对象
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        cross_entropy_m = tf.reduce_mean(cross_entropy)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            # 查找check_point
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model restore......")
                # 加载对应的迭代步长
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                IMG_test, LAB_test = sess.run([img_batch_test, label_batch_test])
                test_acc, test_loss = sess.run([accuracy, cross_entropy_m], feed_dict={x: IMG_test, y: LAB_test})
                print("训练%s步之后, 准确率:%s, , 损失值为:%s" % (global_step, test_acc,test_loss))
            else:
                print('未找到对应的模型')
                return
