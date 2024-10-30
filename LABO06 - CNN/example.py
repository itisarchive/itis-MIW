import os

import numpy as np
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import layers
from keras import models
from keras.api.datasets import mnist


def main():
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    all_images = np.concatenate((train_images, test_images))
    all_labels = np.concatenate((train_labels, test_labels))
    all_labels = np.where(all_labels % 2 == 0, 0, 1)
    train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.3,
                                                                            random_state=1)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=3, batch_size=64)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('test_acc = ', test_acc)


if __name__ == '__main__':
    main()
