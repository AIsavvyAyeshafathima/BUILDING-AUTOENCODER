#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:09:17 2024

@author: ayeshafathima
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load Fashion-MNIST dataset
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.  # Normalize the data to [0, 1] range
x_test = x_test.astype('float32') / 255.

x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension (28, 28, 1)
x_test = np.expand_dims(x_test, axis=-1)

# Build the encoder
encoder = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu')  # Latent space representation
])

# Build the decoder
decoder = models.Sequential([
    layers.InputLayer(input_shape=(64,)),
    layers.Dense(7 * 7 * 128, activation='relu'),
    layers.Reshape((7, 7, 128)),
    layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')  # Output layer
])

# Combine encoder and decoder to form autoencoder
autoencoder = models.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
history = autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, validation_data=(x_test, x_test))

# Test and reconstruct images
reconstructed_images = autoencoder.predict(x_test[:5])

# Plot original and reconstructed images
for i in range(5):
    # Original image
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title('Original')

    # Reconstructed image
    plt.subplot(2, 5, i + 6)
    plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
    plt.title('Reconstructed')

plt.show()
