import os
import numpy as np
from sklearn.cluster import KMeans

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras import models, layers
from keras.api.datasets import mnist

def create_autoencoder(input_shape):
    encoder_layers = [
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(4, (3, 3), activation='relu', padding='same')
    ]

    decoder_layers = [
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same'),
        layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')
    ]

    encoder = models.Sequential(encoder_layers, name="encoder")
    decoder = models.Sequential(decoder_layers, name="decoder")
    autoencoder = models.Sequential(encoder_layers + decoder_layers, name="autoencoder")

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    autoencoder.summary()
    return encoder, decoder, autoencoder


def main():
    (train_images, train_labels), (_, _) = mnist.load_data()
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255

    encoder, decoder, autoencoder = create_autoencoder(input_shape=(28, 28, 1))
    autoencoder.fit(train_images, train_images, epochs=15, batch_size=64)
    autoencoder.save_weights('autoencoder.weights.h5')

    encoder.load_weights('autoencoder.weights.h5')
    encoder.summary()

    compressed_data = encoder.predict(train_images)
    print(f"Compressed data shape: {compressed_data.shape}")

    a, b, c, d = compressed_data.shape
    flattened_data = compressed_data.reshape(a, b * c * d)
    print(f"Flattened data shape: {flattened_data.shape}")

    kmeans = KMeans(n_clusters=10, random_state=0)
    cluster_labels = kmeans.fit_predict(flattened_data)
    centers = kmeans.cluster_centers_
    print(f"Centers: {centers}")

    Y = np.zeros_like(train_labels)
    for i in range(10):
        mask = (cluster_labels == i)
        Y[mask] = np.bincount(train_labels[mask]).argmax()

    print(f"Predicted labels (Y): {Y}")
    print(f"True labels: {train_labels}")

    for i in range(10):
        print(f"Accuracy for cluster {i}: {np.mean(train_labels[Y == i] == i)}")


if __name__ == '__main__':
    main()
