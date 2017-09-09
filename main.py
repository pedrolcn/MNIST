import pandas as pd
from utils import preprocess
from network import Mnist, build_feed
from utils import TrainBatcher
import tensorflow as tf

# Filepaths
PATH = 'C:/Users/pedro_000/PyProjects/MNIST/Data/'
TRAIN_SET = 'train.csv'
TEST_SET = 'test.csv'

# Some constants
LEARNING_RATE = 1.2e-3
MAX_ITERATIONS = 5000
BATCH_SIZE = 64
VALIDATION_SIZE = 0


def read_data(path, train, test):
    print('Reading CSV Data...')
    train_df = pd.read_csv(path + train)
    test_df = pd.read_csv(path + test)
    print('Data Read\n')

    return train_df, test_df,


def main():
    train_df, test_df = read_data(PATH, TRAIN_SET, TEST_SET)
    (train_images, train_labels), (cv_images, cv_labels) = preprocess(train_df, validation_size=0)

    data = TrainBatcher(train_images, train_labels)
    model = Mnist(batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, kernel=5, filters=[32, 64], fc=1024)
    model.build_graph()

    # Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Starting training...')
        for i in range(MAX_ITERATIONS):
            batch_xs, batch_ys = data.next_batch(BATCH_SIZE)
            train_feed = build_feed(model, batch_xs, batch_ys, 0.5)
            train_eval_feed = build_feed(model, batch_xs, batch_ys, 1.0)

            if i % 100 == 0:
                train_accuracy = model.metrics(train_eval_feed)
                print("step %d, Training accuracy: %g"
                      % (i, train_accuracy))
            model.train_op.run(feed_dict=train_feed)
        print('Training finished')

if __name__ == '__main__':
    main()
