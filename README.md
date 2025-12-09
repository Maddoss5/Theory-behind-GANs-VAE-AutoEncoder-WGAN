# Encoders, Decoders, and Variational Autoencoders (VAE)

## 1. The Basics: Encoders and Decoders

**Encoders** transform data from one form to another. In Deep Learning, they typically project high-dimensional data into a smaller, compressed space known as the **Latent Space**.

For example, a $28 \times 28$ image has 784 input features. An encoder compresses this to a low-dimensional vector (e.g., 2D):
$$(x_1, x_2, \dots, x_{784}) \longrightarrow (z_1, z_2)$$

### The Latent Space
* **Dimensionality:** Higher latent dimensions capture more detail.
* **Semantic Mapping:** In a 2D space for a clothing dataset:
    * Positive $x$ might represent **shirts**.
    * Positive $y$ might represent **pants**.

**Decoders** (in Autoencoders and VAEs) reconstruct the image from this latent space.
* Input: Latent vector $z$.
* Output: Reconstructed image $\hat{x}$.

### Latent Manipulation
Because the latent space is a bottleneck, the model forces features to be organized. We can manipulate this:
* Assume Dimension 1 ($x$-axis) represents hair color (Positive $x$ = Blonde, Negative $x$ = Black).
* If we encode an image, take its latent vector, and add a small value to $x$, the decoded image should look like the original person but with **blonde hair**.

## 2. Architecture Implementation

### Encoder
* **Input:** Image.
* **Layers:** We typically use **Conv2D** layers first to extract edges and patterns. Flattening the image immediately would destroy spatial structural relations.
* **Bottleneck:** After convolutions, we use fully connected (Dense) layers to map features to the latent variables.

### Decoder
* **Input:** Latent vector.
* **Layers:** We use **Conv2DTranspose** (Deconvolution) to upsample the latent vector back to the original image dimensions (e.g., $28 \times 28$).

---

## 3. Problems with Standard Autoencoders

Standard Autoencoders are **deterministic**:
1.  **Fixed Output:** The same input always yields the exact same latent code.
2.  **Discontinuous Space ("Holes"):** The latent space is not continuous.
    * If the model learns a point at $(10,10)$, the area nearby at $(10.5, 9.5)$ might not have been trained at all.
    * Decoding $(10.5, 9.5)$ often results in garbage/noise rather than a valid image variation.

**Solution:** We want the latent space to be continuous and smooth (like a Gaussian distribution) so that *any* point sampled from the space yields a valid image.

---

## 4. Variational Autoencoders (VAE)

To solve the "hole" problem, we make the model **probabilistic**.

### The Encoder Modification
Instead of outputting a single point $z$, the encoder predicts a **probability distribution** for the input. It outputs two vectors:
1.  **Mean ($\mu$)**
2.  **Variance ($\sigma^2$)** (or often log-variance)

$$(\mu, \sigma^2) = \text{Encoder}(x)$$

We want this distribution to resemble a Standard Normal Distribution (Gaussian).

### The Reparameterization Trick
To train the network, we need to sample $z$ from this distribution. However, we cannot backpropagate gradients through a random sampling operation.

**The Trick:** We move the randomness to an independent variable $\epsilon$.
1.  Sample noise $\epsilon \sim \mathcal{N}(0, 1)$ (Mean 0, Std Dev 1).
2.  Calculate $z$ deterministically:
    $$z = \mu + \sigma \cdot \epsilon$$

Now, $\epsilon$ is treated as a constant input, allowing gradients to flow back through $\mu$ and $\sigma$.



### The Loss Function
The VAE loss function has two competing parts:

1.  **Reconstruction Loss (MSE):** Forces the output to look like the input.
2.  **KL Divergence:** Forces the latent distribution to look like a Standard Normal Distribution.

$$L = \underbrace{\text{MSE}(x, \hat{x})}_{\text{Reconstruction}} + \beta \cdot \underbrace{D_{KL}(q(z|x) || \mathcal{N}(0,1))}_{\text{Regularization}}$$

### Why Force a Normal Distribution?
We force the latent space to $\mathcal{N}(0,1)$ (Mean 0, Std Dev 1) for **generative capability**.
* If we allow the model to put data anywhere (e.g., mean = 300), we won't know where to sample from to generate *new* images.
* By forcing a unit Gaussian, we know that to generate a new image, we simply sample random noise from $\mathcal{N}(0,1)$ and feed it to the decoder.

### Calculating KL Divergence
For two Gaussian distributions $P$ and $Q$, there is a closed-form solution. When comparing our Encoder output to a Standard Normal $\mathcal{N}(0,1)$, it simplifies to:

$$D_{KL} = -\frac{1}{2} \sum \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right)$$

*Note: It is possible to map to a Gaussian with different parameters (e.g., $\sigma=5$), but you must adjust the prior in the KL term and the sampling logic accordingly.*

### Beta Annealing
In practice, the KL Divergence term can sometimes overpower the reconstruction loss early in training, causing the model to ignore the input data (Posterior Collapse).

**Technique:** Use a weight $\beta$ for the KL term.
1.  Start with $\beta = 0$ (train as a standard Autoencoder).
2.  Slowly increase $\beta$ (e.g., from 0 to 1) over many epochs.
    * *Example:* Turn $\beta$ from 0.0 to 0.1 or 0.5 starting at epoch 20.
