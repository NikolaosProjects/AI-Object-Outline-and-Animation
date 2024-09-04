#              ARTIFICIAL INTELLIGENCE PROJECT
# Animating The Outlines Of Objects Using Image Segmentation Models

[---DESCRIPTION---]

This project consists of training and then implementing an image segmentation artificial intelligence model, in order to provide the outline of objects in images. The Fourier Coefficients of these outlines are extracted, converted to rotating vectors in the complex plane, and then animated.

I used my models to animate the outlines of:

1. A Cat
2. A Car
3. An Airplane
4. Bicycle

[---ALGORITHM EXPLANATION---]

The model is loaded in the GPU. Then, through inference, it automatically identifies a specific object in a given image. The model then applies contours on the identified object, as well as on the background, so that the two are clearly seperated. I then use Python's CV2 to determine the borders of these contours.

To animate the trace, I first need analyze it using its Fourier Transform. Using the borders of the outline created by CV2, I create a numpy array which captures this outline as a set of points in the Complex Plane. I take the fourier transform of these points, and thus determine the outline's fourier coefficients and frequencies. I keep only the 200 largest coefficients (largest in terms of magnitude), along with their corresponding frequencies. I display the outline using its complex points to the user, side by side with a graph showing the distribution of the fourier coefficients given their frequencies (only the top 200 of these coefficients are displayed, as described above)

Since these coefficients are objects in the complex plane (form: x + iy), and they all have a unique frequency associated with them, we can treat them as vectors of specific starting points and magnitudes, all rotating with their given frequencies in the complex plane. Using the Python module Manim, I define a vector and its rotation rate using each fourier coefficient. I then place each of the vectors tip to tail, and then make them rotate about the tip of the previous vector (the first vector rotates about the origin), according to their corresponding frequency. The path of the very last vector is traced. This returns an animation of the original trace, using its fourier coefficients.

[---AI MODELS AND TRAINING FILES---]

The model used is: YOLOv8

The dataset used to train these models on Image Segmentation is the COCO 2017 Dataset: https://cocodataset.org/#home

To use the cuda functionality of pytorch in order to train my model I had use linux, as AMD has no cuda support on Windows. The only drivers AMD offers for usage of their GPUs for Machine Learning, are the ROCm drivers (they allow AMD GPUs to utilize cuda). These drivers are only available on Linux. Even on linux though, there is no official support for my specifc GPU from AMD, although its hardware can technically run ROCm. It took 8 different clean installations of linux Ubuntu until I determined the version of Ubuntu and ROCm that would result in a stable OS and usable GPU for AI Training. All other attempts caused my Ubuntu installation to brake, and I had to delete it and start from the beginning. This is a consequence of hardware support, but no official software support for my specific GPU. I have listed my hardware specifications, as well as the exact versions of Ubuntu and ROCm that resulted in a stable environment below.

[---SPECIFICATIONS---]

Software Specifications:

1) OS: Linux Ubuntu 22.04.01
2) Ubuntu Kernel: "6.5.0-18-generic" 
3) Python Version: 3.10.12
4) ROCm Version: 6.0.2.60002-115~22.04

Hardware Specifications:

1) CPU: AMD Ryzen 5 5600
2) GPU: GIGABYTE AORUS AMD RX6750XT 12G VRAM
3) RAM: 16GB
4) MOTHERBOARD: ASUS B550 PRIME
