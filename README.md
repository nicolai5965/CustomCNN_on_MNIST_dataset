# Custom CNN for MNIST Dataset
This repository contains the implementation of a custom Convolutional Neural Network (CNN) for the MNIST dataset using TensorFlow and Keras. As a first-time implementation of a CNN, the aim is to understand the building blocks and the core concepts of a convolutional neural network.

## Overview
The project covers several important steps:

* Importing necessary libraries and modules.
* Loading the MNIST dataset from TensorFlow's datasets.
* Preprocessing and splitting the data into training, validation, and testing datasets.
* Defining custom convolution and max pooling layers.
* Creating the CNN model using Sequential with custom layers.
* Compiling the model using the Adam optimizer and the sparse categorical cross-entropy loss function.
* Training the model with a specified number of epochs.
* Evaluating the model on the test dataset.
* Plotting the loss and accuracy curves, confusion matrix, and feature maps.
## Requirements
The following libraries are required to run the code:

* Numpy
* TensorFlow
* TensorFlow Datasets
* Matplotlib
* Seaborn
* Scikit-learn
## Usage
To train and evaluate the custom CNN model, simply run the provided Python script. The script will load the MNIST dataset, preprocess it, and create a custom CNN model. It will then train the model on the training dataset, evaluate it on the test dataset, and generate the loss and accuracy curves, confusion matrix, and feature maps.

## Building Blocks of a CNN
A Convolutional Neural Network (CNN) is a type of deep learning model specifically designed for image recognition and classification tasks. The building blocks of a CNN typically include the following layers:

* Convolutional Layer: This layer applies convolution operations to the input image, creating a set of feature maps. It helps to learn spatial features from the input data. In this project, custom convolutional layers are created using the CustomConv2D class.

* Activation Function: After the convolutional layer, an activation function is applied to introduce non-linearity into the network. Common activation functions include ReLU (Rectified Linear Unit) and Leaky ReLU. In this project, the ReLU activation function is used.

* Pooling Layer: Pooling layers are used to reduce the spatial dimensions of the feature maps, making the model more computationally efficient and invariant to small translations. Max pooling is a common pooling operation that selects the maximum value from a region of the feature map. In this project, custom max pooling layers are created using the CustomMaxPooling2D class.

* Fully Connected Layer: After a series of convolutional and pooling layers, the output feature maps are flattened and connected to one or more fully connected layers (Dense layers). These layers perform high-level reasoning and classification based on the learned features. In this project, two dense layers are used: one with ReLU activation and another with softmax activation for the final classification.

* Regularization and Normalization: Regularization techniques, such as L2 regularization, can be used to prevent overfitting by penalizing large weights in the model. Batch normalization is another technique used to stabilize training and accelerate convergence. In this project, both L2 regularization and batch normalization are used in the custom convolutional layers.

By understanding and implementing these building blocks, you can create custom CNN architectures tailored to specific tasks and gain deeper insights into the inner workings of convolutional neural networks.
