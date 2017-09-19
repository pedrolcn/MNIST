import pandas as pd
from tools import preprocess, TrainBatcher
from network import Mnist
import tensorflow as tf

# Filepaths
PATH = 'C:/Users/pedro_000/PyProjects/MNIST/Data/'
TRAIN_SET = 'train.csv'
TEST_SET = 'test.csv'

# Some constants
LEARNING_RATE = 1.2e-3
MAX_ITERATIONS = 5000
BATCH_SIZE = 64
VALIDATION_SIZE = 100


def read_data(path, train, test):
    print('Reading CSV Data...')
    train_df = pd.read_csv(path + train)
    test_df = pd.read_csv(path + test)
    print('Data Read\n')

    return train_df, test_df,


def main():
    train_df, test_df = read_data(PATH, TRAIN_SET, TEST_SET)
    (train_images, train_labels), (cv_images, cv_labels) = preprocess(train_df, validation_size=VALIDATION_SIZE)
    cross = {'features': cv_images, 'labels': cv_labels}

    data = TrainBatcher(train_images, train_labels)
    model = Mnist(data, cv_data=cross, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, kernel=5, filters=[32, 64],
                  dropout=0.5, fc=1024)
    model.build_graph()

    # Training
    saver = tf.train.Saver()

    with tf.Session() as sess:
        model.train(sess, 1000, saver)

if __name__ == '__main__':
    main()
