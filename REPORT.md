In this project, we explored the generation of Fiber Orientation Maps (FOMs) using generative adversarial networks. Our implementation focused on two different approaches: a traditional Generative Adversarial Network (GAN) and a Wasserstein GAN (WGAN). Both models were designed to generate 64x64 pixel patches containing RGB color information that encodes the 3D orientation of nerve fibers in brain tissue sections.

Our generator architecture remained consistent across both implementations, utilizing a 100-dimensional latent vector as input. The generator processes this input through dense layers (100 → 128 → 256 _ 16 _ 16) followed by transposed convolutions (256 → 128 → 3 channels), employing batch normalization and ReLU activations throughout. The final layer uses a sigmoid activation to ensure output values fall within the [0,1] range.

While the generator architecture was shared, the discriminator designs differed between the two approaches. The GAN discriminator employs a sigmoid output for binary classification, while the WGAN critic uses a linear output to estimate the Wasserstein distance. Both share a similar convolutional structure, processing the input through layers of increasing depth (3 → 64 → 128 → 256 channels) with LeakyReLU activations.

We trained both models for 30 epochs using the Adam optimizer with β1=0.5 and β2=0.999. For stability, we implemented several training enhancements, including label smoothing in the GAN (using 0.9 for real labels) and Gaussian noise injection (σ=0.05) in both models. The WGAN implementation included weight clipping at ±0.01 and maintained the standard practice of five critic updates per generator update.

To evaluate our models, we computed the Fréchet Inception Distance (FID) score using 1000 generated samples compared against real FOM patches from Vervet1947. The traditional GAN achieved an FID score of 206.76, while the WGAN scored 275.08. These relatively high scores indicate that both models struggled to fully capture the complex distribution of FOM patterns, with the traditional GAN surprisingly outperforming the WGAN despite the latter's theoretical advantages.

Several factors likely contributed to these high FID scores. First, FOMs represent complex 3D fiber orientations through their RGB values, making them inherently more challenging to generate than typical natural images. Additionally, our relatively simple architecture might lack the capacity to capture all the nuances of fiber orientations. Despite implementing various stability measures, the models might have suffered from mode collapse or inadequate convergence during training.

For future improvements, we recommend several enhancements. The architecture could benefit from progressive growing, self-attention layers, or increased network depth. Training optimizations might include experimenting with different learning rates and implementing gradient penalty for the WGAN. Additionally, the data processing pipeline could be enhanced by increasing patch size for better context or implementing specialized normalization techniques for FOM data.

In conclusion, while both models demonstrate the ability to generate FOM-like patterns, the high FID scores suggest significant room for improvement. The traditional GAN showed better performance, but both approaches would benefit from architectural improvements and more sophisticated training strategies to better capture the complex nature of fiber orientation maps. Despite these challenges, our implementation provides a foundation for further research into generating these specialized scientific visualizations.