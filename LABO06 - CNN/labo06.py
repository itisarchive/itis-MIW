import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from keras import layers
from keras import models
from keras.src.datasets import cifar10
from sklearn.model_selection import train_test_split

# bird (2), cat (3), deer (4), dog (5), frog (6), horse (7)
# airplane (0), automobile (1), ship (8), truck (9)
ALIVE_CLASSES = [2, 3, 4, 5, 6, 7]
UNALIVE_CLASSES = [0, 1, 8, 9]


def load_cifar10_binary_classes():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    train_labels = np.where(np.isin(train_labels, ALIVE_CLASSES), 1, 0)
    test_labels = np.where(np.isin(test_labels, ALIVE_CLASSES), 1, 0)

    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    all_images = np.concatenate((train_images, test_images))
    all_labels = np.concatenate((train_labels, test_labels))

    x_train, x_test, y_train, y_test = train_test_split(
        all_images, all_labels, test_size=0.7, random_state=1
    )
    return x_train, x_test, y_train, y_test


def create_cnn_model(conv_layers=1, input_shape=(32, 32, 3)):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    if conv_layers > 1:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    if conv_layers > 2:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    x_train, x_test, y_train, y_test = load_cifar10_binary_classes()

    results = {}

    for conv_layers in [1, 2, 3]:
        print(f"\nTraining model with {conv_layers} convolutional layer(s):")
        model = create_cnn_model(conv_layers=conv_layers, input_shape=(32, 32, 3))
        model.summary()

        model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=64,
            validation_split=0.1,
            verbose=1
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        results[conv_layers] = test_acc
        print(f"Test accuracy with {conv_layers} convolutional layer(s): {test_acc:.4f}")

    optimal_layers = max(results, key=results.get)
    print(
        f"\nOptimal model has {optimal_layers} convolutional layer(s) "
        f"with test accuracy of {results[optimal_layers]:.4f}"
    )


if __name__ == '__main__':
    main()
