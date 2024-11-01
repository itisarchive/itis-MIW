import os

import numpy as np
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import layers
from keras import models
from keras.api.datasets import mnist


def load_mnist_binary_classes():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    all_images = np.concatenate((train_images, test_images))
    all_labels = np.concatenate((train_labels, test_labels))

    all_images = all_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    all_labels = np.where(all_labels % 2 == 0, 0, 1)

    return train_test_split(all_images, all_labels, test_size=0.7, random_state=1)


def create_cnn_model(conv_layers=1, input_shape=(28, 28, 1)):
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
    x_train, x_test, y_train, y_test = load_mnist_binary_classes()

    results = {}

    for conv_layers in [1, 2, 3]:
        print(f"\nTraining model with {conv_layers} convolutional layer(s):")
        model = create_cnn_model(conv_layers=conv_layers)
        model.summary()
        model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        results[conv_layers] = test_acc
        print(f"Test accuracy with {conv_layers} convolutional layer(s): {test_acc:.4f}")

    optimal_layers = max(results, key=results.get)
    print(
        f"\nOptimal model has {optimal_layers} convolutional layer(s) with test accuracy of {results[optimal_layers]:.4f}")


if __name__ == '__main__':
    main()
