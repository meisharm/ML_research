from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Get the MNIST data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
sap_mnist=[]
sap_labels=[]
for i in range(10):
    sap_mnist.append([])
    sap_labels.append([])
for ind in range(len(mnist.train.images)):
    sap_mnist[mnist.train.labels[ind]].append(mnist.train.images[ind])
    sap_labels[mnist.train.labels[ind]].append(mnist.train.labels[ind])

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 128
display_step = 5
ts_sample=10000 # numnber of test samples
# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10 # custom MNIST classes (0-1 digits)
# dropout = 0.75  # Dropout, probability to keep units (disabled for now)
dropout = 0.75

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int32, None)
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)




def conv2d(x, W, b, strides=1):

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):

    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv_net(x, weights, biases, dropout):

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])

    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
def grad_label(sample): # labling sample with grads -old stuff
    sample_vec=[]
    for i in range(0,9):
        sample_vec.append(np.linalg.norm(sess.run(grads_vec, feed_dict={x: [sample], y: [i], keep_prob: 1})))
    return np.argmin(sample_vec)





def qaunt_dif(list_a,list_b):
    q=0
    for i in range(ts_sample):
        if list_a[i] != list_b[i]:
            q=q+1

    return q

def label_list(sample_list):
    label_l=[]
    for sam in sample_list:
        label_l.append(grad_label(sam))
    return label_l

def label_vec(list):
    label_arg=[]
    for l in list:
       label_arg.append(np.argmax(l))
    return label_arg




def cos(a, b, norm_a=None, norm_b=None):
    print('cos_is_called')

    c = 1

    if isinstance(a, list):
        list_a = a
    else:
        list_a = a.tolist()


    if isinstance(b, list):
        list_b = b
    else:
        list_b = b.tolist()
    if norm_a is None:
        norm_a = np.linalg.norm(list_a)
    if norm_b is None:
        norm_b = np.linalg.norm(list_b)
    mul_ab = float(norm_a * norm_b)
    dot_ab = np.dot(list_a, list_b)
    if mul_ab != 0:
        dot_ab = np.dot(list_a, list_b)
        c = dot_ab / mul_ab
    return c





# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 1024 inputs, 2 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
onehot_y = tf.one_hot(y, 10)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=onehot_y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  #AdagradOptimizer
optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(onehot_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#gradients experiment
grads = optimizer2.compute_gradients(cost)
grads_vec = tf.concat([tf.reshape(_[0] , [-1]) for _ in grads], axis=0)


# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:

        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            index = np.random.randint(low=0, high=len(mnist.train.images))
            batch_x.append(mnist.train.images[index])
            batch_y.append(mnist.train.labels[index])    # uncomment for real la
            # batch_y.append(random_labels[index])  # uncomment to have random labels (and comment out the previous)

        # Run optimization op (backprop) and get loss and weights vector
        _, loss, pred_vec, accur = sess.run([optimizer, cost, pred, accuracy],
                                     feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})


        if step % display_step == 0:
            # print cost for each itereation
            print("Iteration: {} of {}    Cost= {:.11f}  ".format(step * batch_size, training_iters, loss))
            print("accuracy is: " + str(accur))



        step += 1


    print("Training Finished")
    digit_grad =[]
    digit_grad_norm=[]
    for i in range(10):
        grad = sess.run(grads_vec, feed_dict={x: sap_mnist[i], y: sap_labels[i], keep_prob: 1}).tolist()
        digit_grad.append(grad)
        digit_grad_norm.append(np.linalg.norm(grad))
    sam_prob_list=[[] for i in range((ts_sample))]
    for ind in range(ts_sample):
        for dig in range(10):
            sam_prob_list[ind].append(cos(digit_grad[dig],sess.run(grads_vec, feed_dict={x: [mnist.test.images[ind]], y: [dig], keep_prob: 1}),digit_grad_norm[dig]))
    pred_cos=[]

    for ind in range(ts_sample):
     pred_cos.append(np.argmax(sam_prob_list[ind]))

   # pred_ts=label_list(mnist.test.images[:ts_sample])
    net_pred_ts = (conv_net(mnist.test.images[:ts_sample], weights, biases, keep_prob))
    pred_vec=sess.run(net_pred_ts,feed_dict={ keep_prob: 1})
    pred_vec=label_vec(pred_vec)
    print (str(pred_vec))
    #mistakes_grad=qaunt_dif(pred_ts,mnist.test.labels[:ts_sample])
    mistakes_net=qaunt_dif(pred_vec,mnist.test.labels[:ts_sample])
    mistakes_cos=qaunt_dif(pred_cos,mnist.test.labels[:ts_sample])

    print(" ""mistakes_net:  " + str(mistakes_net)+" ""mistakes_cos:  " + str(mistakes_cos)+" ")

