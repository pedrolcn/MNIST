import tensorflow as tf
from tools import weight_variable, bn_variable, conv2d, max_pool_2x2, batch_norm, bias_variable

""" Define the network and model class"""


class Network(object):
    """
    Abstract wrapper class for TensorFlow models, defines the interfaces a network shall have
    """
    def __init__(self, train_data, cv_data=None, batch_size=None, learning_rate=None, kernel=None, filters=None,
                 dropout=None, fc=None):
        self.train_data = train_data

        if cv_data:
            self._cross_validate = True
            self.cv_data = cv_data
        else:
            self._cross_validate = False

        # Hyperparams
        self.batch_size = batch_size
        self.lr = learning_rate
        self.kernel = kernel
        self.filters = filters
        self.dropout = dropout
        self.fc = fc
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self._phase_train = True

    @property
    def phase_train(self):
        return self._phase_train

    @property
    def cross_validate(self):
        return self._cross_validate

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


class Mnist(Network):
    """
    Model for predicting handwritten digit from the MNIST Database. The network implemented is very simillar to the
    TensorFlow tutorial
    """
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
                self.W_fc = weight_variable([64, 10], name='readout_weight')
                self.b_fc = bias_variable([10], name='bias_reaout')

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
                self.h_avg_pool = tf.nn.avg_pool(self.h_pool2, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1],
                                                 padding='VALID', name='Avg_pool')
                self.h_drop = tf.nn.dropout(self.h_avg_pool, keep_prob=self.keep_prob, name='Dropout')

            with tf.name_scope('Readout'):
                self.h_drop_flat = tf.reshape(self.h_drop, [-1, 64])
                self.y_conv = tf.matmul(self.h_drop_flat, self.W_fc) + self.b_fc

    def _create_loss(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))

    def _create_optimizer(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Trainer'):
                # Adam optimizer as usual
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
                                                                         global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope('Summaries'):
            tf.summary.scalar('Loss', self.loss)
            tf.summary.histogram('Histogram Loss', self.loss)

            self.summary_op = tf.summary.merge_all()

    def build_feed(self, batch):
        if self._phase_train:
            feed_dict = {self.x: batch['features'], self.y_: batch['labels'], self.keep_prob: self.dropout}
        else:
            feed_dict = {self.x: batch['features'], self.y_: batch['labels'], self.keep_prob: 1.0}

        return feed_dict

    def metrics(self, batch):
        if self._phase_train:
            self._phase_train = False
            feed_dict = self.build_feed(batch)
            self._phase_train = True
        else:
            feed_dict = self.build_feed(batch)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1)), tf.float32))
        return accuracy.eval(feed_dict)

    def predict(self, feed_dict):
        predict = tf.argmax(self.y_conv, 1)
        return predict.eval(feed_dict)

    def train(self, sess, num_iter, saver=None, logging=100, saving=1000):
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('C:\\Users\\pedro_000\\PyProjects\\MNIST\\logdir')
        train_writer.add_graph(graph=tf.get_default_graph())

        print('Starting training...')

        for i in range(num_iter):
            batch = self.train_data.next_batch(self.batch_size)
            train_feed = self.build_feed(batch)

            if i % logging == 0:
                train_accuracy = self.metrics(batch)

                if self.cross_validate:
                    cv_accuracy = self.metrics(self.cv_data)
                    print("step %d, Training accuracy: %g, CV accuracy: %g" % (i, train_accuracy, cv_accuracy))
                else:
                    print("step %d, Training accuracy: %g" % (i, train_accuracy))

                summary = sess.run(self.summary_op, feed_dict=train_feed)
                train_writer.add_summary(summary, global_step=i)

            if i % saving == 0:
                saver.save(sess, './checkpoints/chck', global_step=self.global_step)

            self.train_op.run(feed_dict=train_feed)
        print('Training finished')
