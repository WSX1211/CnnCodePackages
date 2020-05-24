# encoding=utf-8
__author__ = "WSX"
import tensorflow as tf
import os
from creatDataset import read_and_decode
#同来进行网络训练的代码
learning_rate = 1e-4
train_batch_size = 32
test_batch_size = 1000
STEPS = 100
max_acc = 0  # 最高测试准确率测试
MODEL_SAVE_PATH = "./model/letnet5"
MODEL_NAME = "LetNet5"
graph_path = "./dirs"  #保存graph的文件夹
dataset = "./datasets/dataset1/"

def train(net, IMG_W, IMG_H, CHANNELS, outs):  #outs 表示输出类别个数
    x = tf.placeholder(tf.float32, [-1, IMG_W, IMG_H, CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, outs])
    y = net.build(x)     #网络输出

    global_step = tf.Variable(0, trainable=False)  #全局步长
    #损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy_m = tf.reduce_mean(cross_entropy)
    #优化器
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_m, global_step=global_step)
    #z准确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    #保存模型前需要进行定义
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    #读取数据部分
    images, labels = read_and_decode(dataset + 'train.tfrecords')
    img_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                    batch_size=train_batch_size,
                                                    capacity=392,
                                                    min_after_dequeue=200)
    images_test, labels_test = read_and_decode(dataset + 'test.tfrecords')
    img_batch_test, label_batch_test = tf.train.shuffle_batch([images_test, labels_test],
                                                    batch_size=test_batch_size,
                                                    capacity=392,
                                                    min_after_dequeue=200)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #写入graph
        train_writer = tf.summary.FileWriter(graph_path, sess.graph)
        coord = tf.train.Coordinator()  # 线程协调器
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        #加载之前的模型
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restore......")
        # 开始训练
        for i in range(STEPS):
            IMG, LAB = sess.run([img_batch, label_batch])# 每次取一批次的数据和标注
            sess.run(train_step, feed_dict={x: IMG, y: LAB})
            train_acc, train_loss = sess.run([accuracy, cross_entropy_m], feed_dict={x: IMG, y: LAB})
            print("训练： 迭代第%s次， 准确率：%s, 损失值：%s" %(i,train_acc,train_loss))
            #每五次 测试集测试一下
            if (i % 5) == 0:
                print("训练： 迭代第%s次"%i)
                IMG_test, LAB_test = sess.run([img_batch_test, label_batch_test])
                test_acc, test_loss = sess.run([accuracy, cross_entropy_m], feed_dict={x: IMG_test, y: LAB_test})
                print("测试： 迭代第%s次， 准确率：%s, 损失值：%s" % (i, test_acc, test_loss))
                #构建图
                summay = sess.run(merged, feed_dict={x: IMG_test, y: LAB_test})
                train_writer.add_summary(summay, i)

                if max_acc < test_acc:  # 记录测试准确率最大时的模型
                    max_acc = test_acc
                    saver.save(sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        train_writer.close()
        coord.request_stop()
        coord.join(threads)
        sess.close()

if __name__ == "__main__":
    train( )  #net, IMG_W, IMG_H, CHANNELS, outs