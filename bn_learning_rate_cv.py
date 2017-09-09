import pandas as pd
# Tensorflow and Numpy are loaded from utils
from utils import *

# Filepaths
PATH = 'C:/Users/pedro_000/PyProjects/MNIST/Data/'
TRAIN_SET = 'train.csv'
TEST_SET = 'test.csv'

# Some constants
LEARNING_RATE = 3e-3
MAX_ITERATIONS = 5000
BATCH_SIZE = 64
VALIDATION_SIZE = 0

# Extract training data from CSV file
print('Reading data from CSV...')
data = pd.read_csv(PATH + TRAIN_SET)
print('Data read successfully')

# Pre-processing on the data
images = data.iloc[:, 1:].values
images = images.astype(np.float)

# Scaling to 0.0 - 1.0 values
images = np.multiply(images, 1.0 / 255.0)

# loading the labels and encoding them in a one-hot vector
labels_flat = data.iloc[:, 0].values.ravel()
labels = dense_to_one_hot(labels_flat, 10)

cv_images = images[:VALIDATION_SIZE]
cv_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

# Now we get tensorflowy
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
phase_train = tf.placeholder(tf.bool)

# first convolutional layer
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
conv1_bn = batch_norm(conv2d(x_image, W_conv1), 32, phase_train)

h_conv1 = tf.nn.relu(conv1_bn)
h_pool1 = max_pool_2x2(h_conv1)


# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
conv2_bn = batch_norm(conv2d(h_pool1, W_conv2), 64, phase_train)

h_conv2 = tf.nn.relu(conv2_bn)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# readout layer for deep net
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

# cost function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# optimisation function
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
predict = tf.argmax(y_conv, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# serve data by batches
mnist = TrainBatcher(train_images, train_labels)

# start TensorFlow session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


validation_acc_logger = []
iter_logger = []

print('Starting training...')
for i in range(MAX_ITERATIONS):
    batch_xs, batch_ys = mnist.next_batch(BATCH_SIZE)

    if i % 50 == 0:
        perm = np.random.permutation(VALIDATION_SIZE)[:1000]
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_xs, y_: batch_ys, phase_train: True})
        cv_accuracy = accuracy.eval(feed_dict={
            x: cv_images[perm], y_: cv_labels[perm], phase_train: False})

        print("step %d, Training accuracy: %g | Validation Accuracy: %g"
              % (i, train_accuracy, cv_accuracy))

        iter_logger.append(i)
        validation_acc_logger.append(cv_accuracy)

    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, phase_train: True})
print('Training finished')

print('Logging data...')
# Saves to csv_file
filename = 'lr=' + str(LEARNING_RATE) + '.csv'
np.savetxt(filename, np.c_[iter_logger, validation_acc_logger],
           delimiter=',', header='', comments='', fmt='%f')
sess.close()
