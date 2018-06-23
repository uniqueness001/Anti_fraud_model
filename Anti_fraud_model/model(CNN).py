# coding:utf-8
# @author:zee(GDUT)
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import data_propressing
data_new = data_propressing.processing()
# 实现20列和21列之间的交换
# data_new[:, [-1, -2]] = data_new[:, [-2, -1]]
# print(data_new.head())
x_feature = list(data_new.columns)
x_feature.remove('Class')
X = data_new[x_feature]
y = data_new['Class']
# x_feature = list(data_new.columns)
# x_feature.remove('Class')
# x_val = data_new[x_feature]
# y_val = data_new['Class']
# x_val = data_new[x_feature]
# y_val = data_new['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# train_data =[X_train,y_train]
# test_data = [X_test,y_test]
# sep =int(0.7*len(data_new))
# train_data = data_new[:sep]
# test_data = data_new[sep:]
# 定义神经网络层
def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs
# 定义预测结果的精确值
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result
# 神经网络的输入
xs = tf.placeholder(tf.float32, [None, 19])
ys = tf.placeholder(tf.float32, [None, 1])
# 神经网络中枢
prediction = add_layer(xs, 19, 1,  activation_function=tf.nn.softmax)
# the error between prediction and real data
# 计算loss值
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))
# 训练器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 会话控制，运行的控制
sess = tf.Session()
# 初始化
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
    if i % 50 == 0:
        print(compute_accuracy(
            X_test, y_test))

