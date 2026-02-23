# Variational Autoencoder (VAE) Experiment

![network](./images/ZN6MyTx.png)

## Overview

This experiment demonstrates the implementation of a **Variational Autoencoder (VAE)** using TensorFlow and Keras to generate and reconstruct handwritten digits from the MNIST dataset. Unlike traditional autoencoders that learn a fixed encoding, VAEs learn a probabilistic latent space, enabling the generation of diverse and continuous samples.

## Conceptual Foundation

A VAE bridges the gap between deep learning and probabilistic modeling. It consists of two main distinct networks:

1.  **Encoder**: Maps input images to a latent distribution defined by a mean ($\mu$) and a log-variance ($\log(\sigma^2)$).
2.  **Decoder**: Samples from this distribution to reconstruct the original image.

A key component is the **Reparameterization Trick**, which allows the model to backpropagate gradients through the stochastic sampling process by introducing a random noise variable $\epsilon$.

## Dataset

- **Name**: [MNIST](http://yann.lecun.com/exdb/mnist/) (Modified National Institute of Standards and Technology database)
- **Content**: 60,000 training images and 10,000 testing images of handwritten digits (0-9).
- **Preprocessing**:
    - Images are normalized to the `[0, 1]` range.
    - Reshaped to `(28, 28, 1)` to be compatible with Convolutional layers.

## Architecture

### 1. Encoder

The encoder compresses the input image into a compressed latent representation.

- **Input**: 28x28x1 Grayscale Image.
- **Layers**:
    - Two `Conv2D` layers with `ReLU` activation and stride 2 for downsampling.
    - `Flatten` layer followed by a `Dense` layer (16 units).
    - Two separate `Dense` layers outputting `z_mean` and `z_log_var`.
- **Output**: Latent distribution parameters.

### 2. sampling Layer

- Uses the reparameterization trick: $z = \mu + e^{0.5 \cdot \log(\sigma^2)} \cdot \epsilon$
- Where $\epsilon$ is drawn from a standard normal distribution.

### 3. Decoder

The decoder attempts to reconstruct the image from the latent sample $z$.

- **Input**: Latent vector $z$.
- **Layers**:
    - `Dense` layer reshaped to spatial dimensions `(7, 7, 64)`.
    - Two `Conv2DTranspose` layers with `ReLU` activation and stride 2 for upsampling.
    - Final `Conv2DTranspose` layer with `sigmoid` activation.
- **Output**: 28x28x1 Reconstructed Image.

## Loss Function

The VAE is trained to minimize a combined loss function:

1.  **Reconstruction Loss** (Binary Cross-Entropy): Measures how well the decoder reconstructs the input image.
2.  **KL Divergence Loss**: Regularizes the latent space by measuring the divergence between the learned latent distribution and a standard normal distribution.

## Hyperparameters

- **Latent Dimension**: 2 (Chosen to allow for easy 2D visualization)
- **Batch Size**: 128
- **Epochs**: 15
- **Optimizer**: Adam

## Results & Visualization

The notebook includes a visualization of the **Latent Space Clustering**. By projecting the test set into the 2D latent space, we can observe how different digits form distinct clusters, demonstrating the VAE's ability to learn a meaningful representation of the data.

![Latent Space Clustering](./images/latent_space_clustering.png)

## Libraries Used

- `tensorflow` (Keras)
- `numpy`
- `matplotlib`
