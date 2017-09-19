import numpy as np
import pandas as pd


class DataBatcher(object):
    """
    Loads Data from either url or filepath and serves it into batches. Uses pandas high-level API to handle
    I/O operations, has options to allow for loading data into local storage or loading from buffer
    """
    def __init__(self, url_or_filepath, validation_size=0, num_rows=None, iterate=True, shuffle=True):
        self.validation_size = validation_size
        self.path = url_or_filepath
        self.iterate = iterate
        self.shuffle = shuffle
        self.epoch = 0
        self.index_in_epoch = 0

        if not self.iterate:
            self.df = pd.read_csv(self.path)
            self.num_examples = self.df.shape[0]
        elif not num_rows:
            raise AttributeError("When reading the dataset iteratively the number of rows must be given")
        else:
            self.num_examples = num_rows

        self.indexer = np.arange(self.num_examples)

    def next_batch(self, batch_size):
        """
        Serves data by batches of a given size
        :param batch_size:      the number of items in the batch to be served
        :return: batch          a dict containing the fields 'features' and 'labels'
        """
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        self.epoch += batch_size/self.num_examples

        # When all the training data is ran, shuffles it
        if self.index_in_epoch > self.num_examples and self.shuffle:
            self.indexer = np.random.permutation(self.num_examples)
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.num_examples

        if self.iterate:
            batch_df = pd.DataFrame()
            if self.epoch < 1:
                batch_df = pd.read_csv(self.path, nrows=batch_size, skiprows=start)
            else:
                for i in range(batch_size):
                    item = pd.read_csv(self.path, nrows=1, skiprows=self.indexer[start+i])
                    batch_df = pd.concat(item)
        else:
            batch_df = self.df[start: self.index_in_epoch]

        examples = np.multiply(batch_df.iloc[:, 1:].values.astype(np.float), 1.0 / 255.0)
        labels = self.dense_to_one_hot(batch_df.iloc[:, 0].values.ravel(), 10)

        batch = {'features': examples, 'labels': labels}
        return batch

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
