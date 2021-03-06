import tensorflow as tf
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import classification_image_gen
from classification_image_gen import make_classification_images as make_images
import numpy as np

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,8]))
b = tf.Variable(tf.zeros([8]))

y_ = tf.placeholder(tf.float32, [None, 8])

y = tf.nn.softmax(tf.matmul(x, W)+b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for _ in range(1000):
    batch = make_images()
    train_step.run(feed_dict = {x:batch[0], y_:batch[1]})

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,28])
b_conv2 = bias_variable([28])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([1372,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 1372])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 8])
b_fc2 = bias_variable([8])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

labels = []
predictions = []

for i in range(150):
  batch = make_images()
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  for _ in xrange(99):
      labels.append(batch[1][_])
      predictions.append(y_conv)

  batchtest = make_images(1000)
  if i%1000 ==0:
    test_accuracy = accuracy.eval(feed_dict={x:batchtest[0], y_:batchtest[1], keep_prob: 1.0})
    print("step %d, test accuracy %g"%(i, test_accuracy))

batch = make_images(1000)
print("test accuracy %g"%accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}))

labelvals = []
predictionvals = []

for z in range(len(labels)):
    labelvals.append(np.where(labels[z] == 1))

for q in range(len(predictions)):
    predictionvals.append(np.where(predictions[q]==1))

print labels[:10], predictions[:10]

print labelvals[:10]

fout=open('commonerrors.csv','w')
fout.write('event, label, prediction')
for x in xrange(len(labels)):
    fout.write('%d, %d' % (x, labelvals[x]), predictionvals[x])
fout.close()


#vals = np.zeros(len(mistakes))

#for z in range(len(mistakes)):
 #   if mistakes[z]== [1,0,0,0,0,0,0,0]:
  #      vals[z] = 1.
  #  if mistakes[z] == [0,1,0,0,0,0,0,0]:
   #     vals[z] = 2.
   # if mistakes[z] == [0,0,1,0,0,0,0,0]:
   #     vals[z] =3.
   # if mistakes[z] == [0,0,0,1,0,0,0,0]:
   #     vals[z] = 4.
   # if mistakes[z] == [0,0,0,0,1,0,0,0]:
   #     vals[z] = 5.
   # if mistakes[z] == [0,0,0,0,0,1,0,0]:
   #     vals[z] = 6.
   # if mistakes[z] == [0,0,0,0,0,0,1,0]:
   #     vals[z] = 7.
   # if mistakes[z] == [0,0,0,0,0,0,0,1]:
   #     vals[z] = 8.

#errs, counts = np.unique(vals, return_counts=True)

#tvals = np.zeros(len(totals))

#for z in range(len(totals)):
 #   if totals[z]== [1,0,0,0,0,0,0,0]:
  #      tvals[z] = 1.
   # if totals[z] == [0,1,0,0,0,0,0,0]:
    #    tvals[z] = 2.
   # if totals[z] == [0,0,1,0,0,0,0,0]:
    #    tvals[z] =3.
   # if totals[z] == [0,0,0,1,0,0,0,0]:
    #    tvals[z] = 4.
   # if totals[z] == [0,0,0,0,1,0,0,0]:
    #    tvals[z] = 5.
   # if totals[z] == [0,0,0,0,0,1,0,0]:
    #    tvals[z] = 6.
    #if totals[z] == [0,0,0,0,0,0,1,0]:
    #    tvals[z] = 7.
    #if totals[z] == [0,0,0,0,0,0,0,1]:
     #   tvals[z] = 8.

#terrs, tcounts = np.unique(tvals, return_counts = True)
#print terrs, tcounts
#print errs, counts

#fracs = []
#for z in range(len(counts)):
 #   fracs.append(counts[z]/tcounts[z])
#y_post = np.arange(len(errs))
#plt.bar(y_post, fracs, align = 'center')
#plt.xticks(y_post, errs)
#plt.ylabel("fraction mistakenly labeled")
