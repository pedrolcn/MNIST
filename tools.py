"""Collection of helper functions & utilities for classifying the MNIST dataset"""
import tensorflow as tf
import numpy as np


def bn_variable(init, shape, name=None):
    return tf.Variable(tf.constant(init, shape=[shape]), name=name)


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def batch_norm(x, beta, gamma, phase_train):
    """
    Batch normalization on convolutional maps.
    Slightly modified to accept the beta and gamma as parameters instead of declarin them inside the function
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        beta:        tf.variable, batch normalization beta parameter
        gamma:       tf.variable, batch normalizaation gamma parameter
        x:           Tensor, 4D BHWD input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(tf.cast(phase_train, tf.bool), mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


class TrainBatcher(object):

    def __init__(self, examples, labels):
        self.labels = labels
        self.examples = examples
        self.index_in_epoch = 0
        self.num_examples = examples.shape[0]

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size

        # When all the training data is ran, shuffles it
        if self.index_in_epoch > self.num_examples:
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.examples = self.examples[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples
        end = self.index_in_epoch
        batch = {'features': self.examples[start:end], 'labels': self.labels[start:end]}
        return batch


def preprocess(dataframe, train=True, validation_size=0):
    if train:
        images = np.multiply(dataframe.iloc[:, 1:].values.astype(np.float), 1.0 / 255.0)
        labels = dense_to_one_hot(dataframe.iloc[:, 0].values.ravel(), 10)

        cv_images = images[:validation_size]
        cv_labels = labels[:validation_size]
        train_images = images[validation_size:]
        train_labels = labels[validation_size:]

        return (train_images, train_labels), (cv_images, cv_labels)
    else:
        images = np.multiply(dataframe.iloc[:, 1:].values.astype(np.float), 1.0 / 255.0)
        return images


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot