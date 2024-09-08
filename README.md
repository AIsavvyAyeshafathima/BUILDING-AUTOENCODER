
# BUILDING AUTOENCODER USING FASHION-MNIST
**Introduction**

The objective of this assignment is to design an autoencoder model that can reconstruct images from the Fashion-MNIST dataset. An autoencoder consists of two main parts: an encoder, which compresses the input image into a latent space representation, and a decoder, which reconstructs the image from this compressed representation. This assignment will focus on building, training, and evaluating an autoencoder using convolutional layers.

**Dataset: Fashion-MNIST**

The Fashion-MNIST dataset is a popular dataset consisting of grayscale images, each sized 28x28 pixels, of different fashion items such as shirts, trousers, shoes, and bags. There are 10 classes in the dataset, but for this assignment, we will not use the labels since the task is unsupervised. The training set contains 60,000 images, and the test set contains 10,000 images.
Model Architecture

The autoencoder consists of two parts:

**1. Encoder**

The encoder compresses the input image into a latent space by reducing its spatial dimensions using convolutional layers and max pooling. The architecture of the encoder is as follows:

•	A convolutional layer with 32 filters, a kernel size of 3x3, ReLU activation, and 'same' padding.
•	Max-pooling layer to reduce the spatial dimensions by a factor of 2.
•	A second convolutional layer with 64 filters, a kernel size of 3x3, ReLU activation, and 'same' padding.
•	Another max-pooling layer to further reduce the dimensions.
•	A third convolutional layer with 128 filters, followed by flattening and a dense layer to map the output to a latent vector of size 64.

**2. Decoder**

The decoder reconstructs the compressed latent representation back to the original image size. The architecture of the decoder is:
•	A dense layer to project the latent vector to a 7x7x128 feature map.
•	Reshaping the output to match the feature map shape.
•	Transposed convolutional layers with up sampling to increase the spatial dimensions back to 28x28.
•	A final transposed convolutional layer to output the reconstructed image.

**Results**

The model was trained for 10 epochs, and the following images show the comparison between the original and reconstructed images. For the purpose of this assignment, we selected 5 images from the test set to evaluate the performance of the autoencoder.
Observations:

1.	Image 1: Shoes
o	The general shape of the shoes is well preserved in the reconstructed image.
o	Some finer details, such as the outline, appear slightly blurred.

3.	Image 2: Long-sleeved Shirt
o	The reconstructed image retains the shape and texture of the shirt.
o	However, the 'LE' text on the shirt is less clear compared to the original.

5.	Image 3: Trousers
o	The trousers' shape and size are well reconstructed.
o	Minor details, such as folds, are slightly blurred.

7.	Image 4: Trousers
o	Like Image 3, the overall shape is intact, but small textures are missing.

9.	Image 5: T-shirt
o	The overall structure is preserved, but the texture and depth of the shirt in the original are more prominent than in the reconstruction.
 
The reconstructed images generally capture the essential features of the original images. However, there is a noticeable reduction in the quality of finer details. This is typical behavior for simple autoencoders, which prioritize compressing the most important visual features while sacrificing some of the subtler ones.
 
**Conclusion**

In this assignment, we successfully built an autoencoder using convolutional layers to reconstruct images from the Fashion-MNIST dataset. The encoder compresses the input into a 64-dimensional latent space, while the decoder attempts to reconstruct the original image from this compressed representation. The autoencoder was able to recreate the basic shapes and patterns of the fashion items, but with some loss of detail. This assignment demonstrates how autoencoders can be used for image compression and reconstruction tasks.

