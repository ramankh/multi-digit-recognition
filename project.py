import tensorflow as tf
from helper import GetData
import time
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt


img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10
len_classes = 7
patch_size = 5
k1_size = 8
k2_size = 16
keep_prob = 1
batch_size = 1
train = False
fc_weight_size = 7 * 7 * k2_size

# Creat a GetData class which will load all data from
# specified path for both training and testing data
# Using pickle if hdf5 is false
data = GetData(use_hdf5=False, path="data.pickle")
test_data = GetData(use_hdf5=False, path="test_data.pickle")

sess = tf.InteractiveSession()


# Get a batch of data
# x represent image data in a flatten array
# Each y represents labels for each digit - this is list of
# one hot arrays which each array has only one item = 1


x_ = tf.placeholder(tf.float32, shape=[batch_size, 784])
y_one_ = tf.placeholder(tf.int32, shape=[batch_size])
y_two_ = tf.placeholder(tf.int32, shape=[batch_size])
y_three_ = tf.placeholder(tf.int32, shape=[batch_size])
y_four_ = tf.placeholder(tf.int32, shape=[batch_size])
y_five_ = tf.placeholder(tf.int32, shape=[batch_size])
y_len_ = tf.placeholder(tf.int32, shape=[batch_size])

#Initiate convolutions weight variable with Xavier initializer
def convs_weight(shape, name):
	return tf.get_variable(shape=shape,
		initializer=tf.contrib.layers.xavier_initializer_conv2d(), name=name)

#returns a variablae for bias
def convs_bias(shape):
  	return tf.Variable(tf.constant(1.0, shape=shape))

#returns initaiated weight variable with xavier initializer
#for fully connecter layers
def fc_weight(shape, name):
	return tf.get_variable(shape=shape,
           initializer=tf.contrib.layers.xavier_initializer(), name=name)

#returns bias variable
#for fully connecter layers
def fc_bias(shape):
	return tf.Variable(tf.constant(1.0, shape=shape))

#return convloutional layer with strides = 1 and same padding
#with input = x and weigts = W
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#returns a maxpooling layer with window size of 2 and same padding
#for input=x
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#create a weight and bias varialble for each fully connecter layer
W_fc_one = fc_weight(shape=[fc_weight_size, num_classes], name='W_F1')
b_fc_one = fc_bias(shape=[num_classes])

W_fc_two = fc_weight(shape=[fc_weight_size, num_classes], name='W_F2')
b_fc_two = fc_bias(shape=[num_classes])

W_fc_three = fc_weight(shape=[fc_weight_size, num_classes], name='W_F3')
b_fc_three = fc_bias(shape=[num_classes])

W_fc_four = fc_weight(shape=[fc_weight_size, num_classes], name='W_F4')
b_fc_four = fc_bias(shape=[num_classes])

W_fc_five = fc_weight(shape=[fc_weight_size, num_classes], name='W_F5')
b_fc_five = fc_bias(shape=[num_classes])

W_fc_len = fc_weight(shape=[fc_weight_size, len_classes], name='W_FL')
b_fc_len = fc_bias(shape=[len_classes])

#crate a scope name to get more modular graph representation in
#tensor board
with tf.name_scope("Layers"):
	#initiate a weight and bias variable for first convolution layer
	#reshape input to four dimensional tensor [1, 28, 28, 1]
	#pass the input to a convolutional layer and send its out put to
	#ReLU activation function
	#max pool the final result of conv one
	#repeat the same process for conv two
	#at the end pass the output to a dropout layer
	#######################  Conv 1  #####################################
	W_conv1 = convs_weight(shape=[patch_size,
		patch_size, num_channels, k1_size], name='W_L1')
	b_conv1 = convs_bias(shape=[k1_size])

	x_image = tf.reshape(x_, [-1, 28, 28, 1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	#imsumm = tf.image_summary("img", x_image, 1000)
	h_pool1 = max_pool_2x2(h_conv1)

	#######################  Conv 2  #####################################
	W_conv2 = convs_weight(shape=[patch_size,
		patch_size, k1_size, k2_size], name='W_L2')
	b_conv2 = convs_bias(shape=[k2_size])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	keep_prob = tf.placeholder(tf.float32)
	h_dropped = tf.nn.dropout(h_pool2, keep_prob)
	shape = h_dropped.get_shape().as_list()
	#h_pool2_flat is the feature variable H discussed in report.pdf
	h_pool2_flat = tf.reshape(
	    h_dropped, [shape[0], shape[1] * shape[2] * shape[3]])

	h_fc1 = tf.matmul(h_pool2_flat, W_fc_one) + b_fc_one
	#tf.histogram_summary('fc1', h_fc1)

	h_fc2 = tf.matmul(h_pool2_flat, W_fc_two) + b_fc_two
	#tf.histogram_summary('fc2', h_fc2)

	h_fc3 = tf.matmul(h_pool2_flat, W_fc_three) + b_fc_three
	#tf.histogram_summary('fc3', h_fc3)

	h_fc4 = tf.matmul(h_pool2_flat, W_fc_four) + b_fc_four
	#tf.histogram_summary('fc4', h_fc4)

	h_fc5 = tf.matmul(h_pool2_flat, W_fc_five) + b_fc_five
	#tf.histogram_summary('fc5', h_fc5)

	h_fcl = tf.matmul(h_pool2_flat, W_fc_len) + b_fc_len
	#tf.histogram_summary('fc_len', h_fcl)

#calculating loss by getting areduced mean of all outputs cross entropy
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc1, y_one_)) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc2, y_two_)) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc3, y_three_)) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc4, y_four_)) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc5, y_five_)) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fcl, y_len_))

loss_summ = tf.scalar_summary('loss', loss)

# minimize the the loss by using the adagrad algorithm
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.93)
train_step = tf.train.AdagradOptimizer(
    learning_rate).minimize(loss, global_step=global_step)
with tf.name_scope("Predictins"):
	pred_one = tf.argmax(tf.nn.softmax(h_fc1), 1)
	pred_two = tf.argmax(tf.nn.softmax(h_fc2), 1)
	pred_three = tf.argmax(tf.nn.softmax(h_fc3), 1)
	pred_four = tf.argmax(tf.nn.softmax(h_fc4), 1)
	pred_five = tf.argmax(tf.nn.softmax(h_fc5), 1)
	pred_len = tf.argmax(tf.nn.softmax(h_fcl), 1)

tf.histogram_summary('pred_one', pred_one)
tf.histogram_summary('pred_two', pred_two)
tf.histogram_summary('pred_three', pred_three)
tf.histogram_summary('pred_four', pred_four)
tf.histogram_summary('pred_five', pred_five)
tf.histogram_summary('pred_len', pred_len)

#storing all prediction in a tf.pack to find the accuracy
#later in correct_predictions
preds = tf.pack([pred_one, pred_two, pred_three, pred_four,
	pred_five])
y_ = tf.pack([tf.cast(y_one_, tf.int64), tf.cast(y_two_, tf.int64)	, tf.cast(
    y_three_, tf.int64), tf.cast(y_four_, tf.int64), tf.cast(y_five_, tf.int64)])

train_correct_prdictions = tf.contrib.metrics.accuracy(
    predictions=preds, labels=y_)
train_summary = tf.scalar_summary('train_accuracy', train_correct_prdictions)

valid_correct_prdictions = tf.contrib.metrics.accuracy(
    predictions=preds, labels=y_)
valid_summary = tf.scalar_summary('valid_accuracy', valid_correct_prdictions)

test_correct_prdictions = tf.contrib.metrics.accuracy(
    predictions=preds, labels=y_)
test_summary = tf.scalar_summary('test_accuracy', test_correct_prdictions)

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
#restoring the model
saver.restore(sess, "saved/model.ckpt")
print("Model restored.")


writer = tf.train.SummaryWriter('logs/', sess.graph)
start_time = time.time()

#train the model if train flag is on
if train == True:
	for i in range(0, 200000):
		#getting the train and valid batch in batch size
		train_batch, valid_batch = data.nextBatch(batch_size)
		_, loss_value = sess.run([train_step, loss_summ], feed_dict={x_: train_batch[0],
			y_one_: train_batch[1], y_two_: train_batch[2], y_three_: train_batch[3], y_four_: train_batch[4], y_five_: train_batch[5],
		 	y_len_: train_batch[6], keep_prob: 0.5})
		writer.add_summary(loss_value, i)
		#check train and valid accuracy each 200 steps
		if i % 200 == 0 and i != 0:
			elapsed = time.time() - start_time

			train_acc, train_summ = sess.run([train_correct_prdictions, train_summary], feed_dict={x_: train_batch[0],
			y_one_: train_batch[1], y_two_: train_batch[2], y_three_: train_batch[3], y_four_: train_batch[4], y_five_: train_batch[5],
		 	y_len_: train_batch[6], keep_prob: 1})
			writer.add_summary(train_summ, i)
			valid_acc, valid_summ = sess.run([valid_correct_prdictions, valid_summary], feed_dict={x_: valid_batch[0],
			y_one_: valid_batch[1], y_two_: valid_batch[2], y_three_: valid_batch[3], y_four_: valid_batch[4], y_five_: valid_batch[5],
		 	y_len_: valid_batch[6], keep_prob: 1})
			writer.add_summary(valid_summ, i)
			start_time = time.time()
			print "step {}, elapsed_time {} training accuracy {} valid accuracy {}".format(i, elapsed, train_acc, valid_acc)
		#check test accuracy every 500 steps
		if i % 500 == 0 and i != 0:
			test_batch = test_data.nextTestBatch(batch_size)
			test_acc, test_summ = sess.run([test_correct_prdictions, test_summary], feed_dict={x_: test_batch[0],
			y_one_: test_batch[1], y_two_: test_batch[2], y_three_: test_batch[3], y_four_: test_batch[4], y_five_: test_batch[5],
		 	y_len_: test_batch[6], keep_prob: 1})
			writer.add_summary(test_summ, i)
			print "step {} test accuracy {}".format(i, test_acc)
		writer.flush()
	#save the model in specified path
	save_path = saver.save(sess, "saved/model.ckpt")
	print("Model saved in file: %s" % save_path)

###############For model evaluetion ####################
#rest of the code are provided for testing the model
#by plotting the filters in conv layers
#for testing the model you need to change the train
#flag in top to false

def getActivations(layer, img):
	print "{}{}{}{}{}{}".format(img[1], img[2], img[3], img[4], img[5], img[6])
	units,predicts,plen=sess.run([layer,preds,pred_len],feed_dict={x_: img[0], y_one_:img[1], y_two_:img[2],y_three_:img[3], y_four_:img[4], y_five_:img[5],y_len_:img[6], keep_prob: 1})
	print "preds is: {}{}{}{}{}{}".format(predicts[0],predicts[1],predicts[2],predicts[3], predicts[4], plen)
	plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(28,28))
    for i in xrange(0,filters):
        plt.subplot(4,4,i+1)
        plt.subplots_adjust(bottom = 0.1, top = 0.9, hspace=0.5)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")

def testStat(runNum):
	for i in range(1000,runNum):
		test_batch = test_data.nextTestBatch(batch_size)
		test_acc, test_summ = sess.run([test_correct_prdictions, test_summary], feed_dict={x_: test_batch[0],\
		y_one_:test_batch[1], y_two_:test_batch[2],y_three_:test_batch[3], y_four_:test_batch[4], y_five_:test_batch[5],\
	 	y_len_:test_batch[6], keep_prob: 1})
		writer.add_summary(test_summ, i)
		print "step {} test accuracy {}".format(i, test_acc)
		writer.flush()

if train == False:
	plt.ion()

	showImg = data.getOne()

	getActivations(h_conv1,showImg)
	raw_input("Press any key to continue")
	getActivations(h_conv2,showImg)
	raw_input("Press any key to exit")
