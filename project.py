import tensorflow as tf
from helper import GetData


img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 11
len_classes = 7
patch_size = 5
k1_size = 8
k2_size = 16
keep_prob = 1
batch_size = 100

fc_weight_size = 784

# Creat a GetData class which will load all data from
# specified path and using pickle if hdf5 is false
data = GetData(use_hdf5 = False, path = "data.pickle")
test_data = GetData(use_hdf5 = False, path = "test_data.pickle")

sess = tf.InteractiveSession()



# Get a batch of data
# x represent image data in a flatten array
# Each y represents labels for each digit - this is list of
# on hot arrays which each array has only one item = 1


x_ = tf.placeholder(tf.float32, shape=[batch_size, 784])
y_one_ = tf.placeholder(tf.int32, shape=[batch_size])
y_two_ = tf.placeholder(tf.int32, shape=[batch_size])
y_three_ = tf.placeholder(tf.int32, shape=[batch_size])
y_four_ = tf.placeholder(tf.int32, shape=[batch_size])
y_five_ = tf.placeholder(tf.int32, shape=[batch_size])
y_len_ = tf.placeholder(tf.int32, shape=[batch_size])

def convs_weight(shape, name):
	return tf.get_variable(shape=shape,\
		initializer=tf.contrib.layers.xavier_initializer_conv2d(), name=name)

def convs_bias(shape):
  	return tf.Variable(tf.constant(1.0, shape=shape))

def fc_weight(shape, name):
	return tf.get_variable(shape=shape,\
           initializer=tf.contrib.layers.xavier_initializer(), name=name)

def fc_bias(shape):
	return tf.Variable(tf.constant(1.0, shape=shape))


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_fc_one = fc_weight(shape = [fc_weight_size, num_classes], name='W_F1')
b_fc_one = fc_bias(shape=[num_classes])

W_fc_two = fc_weight(shape = [fc_weight_size, num_classes], name='W_F2')
b_fc_two = fc_bias(shape=[num_classes])

W_fc_three = fc_weight(shape = [fc_weight_size, num_classes], name='W_F3')
b_fc_three = fc_bias(shape=[num_classes])

W_fc_four = fc_weight(shape = [fc_weight_size, num_classes], name='W_F4')
b_fc_four = fc_bias(shape=[num_classes])

W_fc_five = fc_weight(shape = [fc_weight_size, num_classes], name='W_F5')
b_fc_five = fc_bias(shape=[num_classes])

W_fc_len = fc_weight(shape = [fc_weight_size, len_classes], name='W_FL')
b_fc_len = fc_bias(shape=[len_classes])

with tf.name_scope("layers"):
	#######################  Conv 1  #####################################
	W_conv1 = convs_weight(shape=[patch_size,\
		patch_size,num_channels,k1_size], name='W_L1')
	b_conv1 = convs_bias(shape=[k1_size])

	x_image = tf.reshape(x_, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	#######################  Conv 2  #####################################
	W_conv2 = convs_weight(shape=[patch_size,\
		patch_size,k1_size,k2_size], name='W_L2')
	b_conv2 = convs_bias(shape=[k2_size])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# Flatten feature output of last conv + pooling layer
	#h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*k2_size])
	keep_prob = tf.placeholder(tf.float32)
	h_dropped = tf.nn.dropout(h_pool2, keep_prob)
	shape = h_dropped.get_shape().as_list()
	h_pool2_flat = tf.reshape(h_dropped,[shape[0], shape[1]*shape[2]*shape[3]])

	h_fc1 = tf.matmul(h_pool2_flat, W_fc_one) + b_fc_one
	tf.histogram_summary('fc1', h_fc1)
	h_fc2 = tf.matmul(h_pool2_flat, W_fc_two) + b_fc_two

	h_fc3 = tf.matmul(h_pool2_flat, W_fc_three) + b_fc_three

	h_fc4 = tf.matmul(h_pool2_flat, W_fc_four) + b_fc_four

	h_fc5 = tf.matmul(h_pool2_flat, W_fc_five) + b_fc_five

	h_fcl = tf.matmul(h_pool2_flat, W_fc_len) + b_fc_len

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc1, y_one_)) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc2, y_two_)) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc3, y_three_)) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc4, y_four_)) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fc5, y_five_)) +\
tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(h_fcl, y_len_))

tf.scalar_summary('loss', loss)

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.05, global_step, 10000, 0.95)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

pred_one= tf.argmax(tf.nn.softmax(h_fc1),1)
pred_two =  tf.argmax(tf.nn.softmax(h_fc2),1)
pred_three =  tf.argmax(tf.nn.softmax(h_fc3),1)
pred_four =  tf.argmax(tf.nn.softmax(h_fc4),1)
pred_five =  tf.argmax(tf.nn.softmax(h_fc5),1)
pred_len =  tf.argmax(tf.nn.softmax(h_fcl),1)

preds = tf.pack([pred_one, pred_two, pred_three, pred_four, \
	pred_five])
y_ = tf.pack([tf.cast(y_one_,tf.int64), tf.cast(y_two_,tf.int64)\
	, tf.cast(y_three_,tf.int64), tf.cast(y_four_,tf.int64), tf.cast(y_five_,tf.int64)])

correct_prdictions = tf.contrib.metrics.accuracy(predictions=preds, labels= y_)

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
#saver.restore(sess, "saved/model.ckpt")
#print("Model restored.")
writer = tf.train.SummaryWriter('logs/', sess.graph)

for i in range(10):
  #x, y_one, y_two, y_three, y_four, y_five, y_len = data.nextBatch2(batch_size)
  x, y_one, y_two, y_three, y_four, y_five, y_len = data.getHun(batch_size)
  t_x, t_y_one, t_y_two, t_y_three, t_y_four, t_y_five, t_y_len = test_data.nextBatch2(batch_size)

  train_step.run(feed_dict={x_: x, y_one_:y_one, y_two_:y_two,\
  y_three_:y_three, y_four_:y_four, y_five_:y_five, y_len_:y_len, keep_prob: 0.5})

  if i%2 == 0:
	train_accuracy = correct_prdictions.eval(feed_dict={x_: x, y_one_:y_one, \
		y_two_:y_two, y_three_:y_three, y_four_:y_four, y_five_:y_five, y_len_:y_len, keep_prob: 1.0})
	print pred_one
	print("step %d, training accuracy %g"%(i, train_accuracy))

test_accuracy = correct_prdictions.eval(feed_dict={x_: t_x, y_one_:t_y_one, \
	y_two_:t_y_two, y_three_:t_y_three, y_four_:t_y_four, y_five_:t_y_five, y_len_:t_y_len, keep_prob: 1.0})
print("step %d, test accuracy %g"%(i, test_accuracy))

save_path = saver.save(sess, "saved/model.ckpt")
print("Model saved in file: %s" % save_path)