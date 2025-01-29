# Autoencoder_MNIST_Classifier
![Autoencoder](https://github.com/user-attachments/assets/30f9c6e6-7bed-4024-b1fb-d8534347a9a8)

## Autoencoder Feature Extraction with MNIST

This project demonstrates the use of autoencoders for feature extraction, showing how an **unsupervised model** can be used to learn effective features from data. 

### Project Overview

In this fun little experiment, I trained a simple autoencoder on the MNIST dataset, with the goal of learning representations (latent features) of the input images. Once the autoencoder was trained, the encoder part was frozen, and its output was passed to a shallow classifier. This shallow classifier was able to achieve **over 96% accuracy** within just **5 epochs**.

### Key Steps
1. **Autoencoder Training**: A simple convolutional autoencoder was trained on the MNIST dataset to learn latent features.
2. **Freezing the Encoder**: After training, the encoder was frozen, retaining its learned features while allowing the classifier to be trained on top of them.
3. **Shallow Classifier**: The output from the encoder was used as input for a shallow classifier, resulting in excellent performance within a few epochs.

### Why This is Fun and Useful
- **Efficient Feature Learning**: Autoencoders help in creating efficient, compressed representations of the data, which can significantly reduce computational complexity for downstream tasks.
- **Great for Real-World Applications**: This method of feature extraction can be applied to more complex datasets, improving the performance of classifiers even with limited data.

Feel free to explore the code and see how the autoencoder learns the features — and how quickly the classifier takes advantage of them!

### Requirements
- Pytorch
- MNIST Dataset 

Enjoy experimenting with this project, and see how well the features generalize to other datasets!

