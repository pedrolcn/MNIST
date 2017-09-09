import tensorflow as tf
from utils import weight_variable, bn_variable, conv2d, max_pool_2x2, batch_norm, bias_variable

""" Define the network and model class"""


class Network(object):
    """
    Abstract wrapper class for TensorFlow models, defintes the interfaces of the network, all the _create methods are
    mere placeholders for the call in the build_graph method, and all shall be overridden in the child classes.
    """

    def __init__(self, batch_size, learning_rate, kernel, filters, fc):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.kernel = kernel
        self.filters = filters
        self.fc = fc
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self._phase_train = True

    @property
    def phase_train(self):
        return self._phase_train

    def _create_placeholders(self):
        raise NotImplementedError

    def _create_variables(self):
        raise NotImplementedError

    def _create_network(self):
        raise NotImplementedError

    def _create_loss(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def _create_summaries(self):
        raise NotImplementedError

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_network()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()


class Model(Network):

    def _create_placeholders(self):
        with tf.name_scope('Inputs'):
            self.keep_prob = tf.placeholder(tf.float32)
            self.x = tf.placeholder(tf.float32, shape=[None, 784])
            self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

    def _create_variables(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Conv1'):
                self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
                self.W_conv1 = weight_variable([self.kernel, self.kernel, 1, self.filters[0]], name='Filters1')
                self.beta_conv1 = bn_variable(0.0, self.filters[0], name='beta_conv1')
                self.gamma_conv1 = bn_variable(1.0, self.filters[0], name='gamma_conv1')

            with tf.name_scope('Conv2'):
                self.W_conv2 = weight_variable([self.kernel, self.kernel, self.filters[0], self.filters[1]],
                                               name='Filters2')
                self.beta_conv2 = bn_variable(0.0, self.filters[1], name='beta_conv2')
                self.gamma_conv2 = bn_variable(1.0, self.filters[1], name='gamma_conv2')

            with tf.name_scope('Readout'):
                self.W_fc = weight_variable([7*7*64, 1024], name='readout_weight')
                self.b_fc = bias_variable([1024], name='bias_reaout')

    def _create_network(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Conv1'):
                self.conv1_bn = batch_norm(conv2d(self.x_image, self.W_conv1), self.beta_conv1,
                                           self.gamma_conv1, self.phase_train)
                self.h_conv1 = tf.nn.relu(self.conv1_bn)
                self.h_pool1 = max_pool_2x2(self.h_conv1)

            with tf.name_scope('Conv2'):
                self.conv2_bn = batch_norm(conv2d(self.h_pool1, self.W_conv2), self.beta_conv2,
                                           self.gamma_conv2, self.phase_train)

                self.h_conv2 = tf.nn.relu(self.conv2_bn)
                self.h_pool2 = max_pool_2x2(self.h_conv2)

            with tf.name_scope('Avg_pool'):
                self.h_avg_pool = tf.nn.avg_pool(self.h_pool2, ksize=[1, 1, 1, 64], strides=[1, 1, 1, 1],
                                                 padding='Same', name='Avg_pool')
                self.h_drop = tf.nn.dropout(self.h_avg_pool, keep_prob=self.keep_prob, name='Dropout')

            with tf.name_scope('Readout'):
                self.y_conv = tf.matmul(self.h_drop, self.W_fc) + self.b_fc

    def _create_loss(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

    def _create_optimizer(self):
        with tf.device('/cpu:0'):
            # Adam optimizer as usual
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                                     global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('Summaries'):
            tf.summary.scalar('Loss', self.loss)
            tf.summary.histogram('Histogram Loss', self.loss)

            self.summary_op = tf.summary.merge_all()
