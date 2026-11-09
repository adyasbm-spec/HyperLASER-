# HyperLASER: Deep Learning-Based MRI Reconstruction

MRI scans provide detailed images of the human body, but acquiring them can take a considerable amount of time. One way to reduce scan time is to collect fewer measurements during the scan. The problem is that using fewer measurements can lead to missing information, resulting in blurry images, noise, and reconstruction artifacts.

HyperLASER is a deep learning-based MRI reconstruction project that explores how these missing details can be recovered from undersampled multi-coil MRI data. It combines the mathematical principles behind SENSE reconstruction with a learned network that progressively refines the reconstructed image.

The model follows an iterative approach. It first estimates coil sensitivity maps from the available calibration data and uses them to obtain an initial image. During reconstruction, it alternates between data consistency steps, which account for the acquired k-space measurements, and a U-Net-based refinement network that learns to improve the image.

One of the main aspects of the implementation is its separate treatment of image magnitude and phase. Instead of refining only the magnitude image, the network predicts corrections for both, allowing the reconstruction process to retain complex-valued MRI information. A learnable sensitivity refinement module is also included to adjust the estimated coil sensitivity maps.

The project is implemented in Python using PyTorch and includes the reconstruction architecture, MRI preprocessing, training, and evaluation components.

### Main Features

- SENSE-based multi-coil MRI reconstruction.
- Learnable sensitivity map refinement.
- Iterative reconstruction with trainable update parameters.
- Separate magnitude and phase refinement.
- U-Net-based image refinement with depthwise separable convolutions.
- Fourier-domain forward and adjoint operators.
- K-space data consistency.
- Training and evaluation using MRI datasets.

### Technology Stack

- Python
- PyTorch
- NumPy
- h5py
- CUDA

### Project Goal

The goal is to study whether combining physics-based reconstruction with learned image refinement can produce reliable MRI reconstructions from undersampled measurements. The project also provides a foundation for experimenting with different reconstruction strategies, network configurations, and acceleration factors.
