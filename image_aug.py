from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from utils import deskew
import pandas as pd
import numpy as np

PATH = 'C:/Users/pedro_000/PyProjects/MNIST/Data/'
TRAIN_SET = 'train.csv'

print('Reading data from CSV...')
data = pd.read_csv(PATH + TRAIN_SET)
print('Data read successfully')

images = data.iloc[:, 1:].values
images = images.astype(np.float)

images = images.reshape(images.shape[0], 28, 28, 1)

for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(images[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    # labels_string = labels_string + ' ' + str(labels_flat[i])
# show the plot
plt.show()

datagen = ImageDataGenerator(preprocessing_function=deskew)
datagen.fit(images)
for X_batch in datagen.flow(images, batch_size=9, shuffle=False):
    for i in range(0, 9):
        plt.subplot(331 + i)
        plt.imshow(X_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))

    plt.show()
    break