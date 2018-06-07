# **Traffic Sign Recognition**

## Objective

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup/images/diff_class_img.jpg "diff_class_img"
[image2]: ./writeup/images/train_hist.jpg "train_hist"
[image3]: ./writeup/images/valid_hist.jpg "valid_hist"
[image4]: ./writeup/images/test_hist.jpg "test_hist"
[image5]: ./writeup/images/test_img.jpg "test_img"
[image6]: ./writeup/images/augment_img.jpg "augment_img"
[image7]: ./writeup/images/train_hist_aug.jpg "train_hist_aug"
[image8]: ./writeup/images/valid_hist_aug.jpg "valid_hist_aug"
[image9]: ./writeup/images/gray_before.jpg "gray_before"
[image10]: ./writeup/images/gray_after.jpg "gray_after"

## Data Set Summary and Exploration

This project uses the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) as input for training a Neural network for classifying German traffic signs

Looking at the dimensions of the provided dataset through simple numpy commands gives us the different sizes for the training, validation, and test sets

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

The 43 unique traffic signs in the training set are shown below. These images were randomly picked from the training data set for each class. From the sample images we can see that the training data set contains a wide variety of input traffic sign images that range in the level of brightness and also orientation

![alt text][image1]

Since there are 43 unique classes it is also important to look at the distribution of the images per class. The histogram plots below show that the data is skewed and certain classes have more training examples than others. This can be addressed through data augmentation to create new training examples

Training set distribution:

![alt test][image2]

Validation set distribution:

![alt test][image3]

Testing set distribution:

![alt test][image4]

## Design and Testing

### Pre-processing

The 3 main pre-processing steps that were performed are:

* Data augmentation
* Grayscale conversion
* Normalization

##### Data augmentation

Data augmentation is performed to increase the overall training examples and to also give a uniform distribution of examples for all the classes since the original distribution is skewed

This was done by setting a minimum number of images per classes (`2000`). New images for a class were created by randomly picking an image from that class and applying a random rotation and brightness adjustment. The rotation was achieved by applying a rotation matrix for random angles between -15 and +15 degrees. The random brightness was achieved by first converting the image to HSV and adding a random value to V before converting back to an RGB image. An example on a training set image before and after augmentation is shown below

Before               |  After
:-------------------:|:-------------------:
![alt test][image5]  |![alt test][image6]

The new images were appended to the training set and a new set of training and validation sets were created using the `train_test_split` function from the sklearn library. The new augmented training set had 77409 examples and the augmented validation set had 8601 examples

Training set distribution before    |  Training set distribution after
:----------------------------------:|:----------------------------------:
![alt test][image2]                 |![alt test][image7]

Validation set distribution before    |  Validation set distribution after
:------------------------------------:|:------------------------------------:
![alt test][image3]                   |![alt test][image8]

##### Grayscale conversion

Grayscale conversion is primilary used to reduce the number of input channels. This may not always be desired as we lose some information by averaging across the RGB channels but it helps speed of the training of the neural network as there are less parameters in the input layer. As can be seen in the results there was not a big impact to the performance

Before grayscale conversion    |  After grayscale conversion
:-----------------------------:|:-----------------------------:
![alt test][image9]            |![alt test][image10]

##### Normalization

Normalization is also done on the images to speed up training. Normalizing the inputs to a neural network speeds up the convergence of the gradient descent algorithm

### Model Architecture

The final model is the LeNet architecture and consisted of the following layers

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized, grayscale image			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	 				|
| Fully connected		| inputs 400, outputs 120						|
| RELU					|												|
| Fully connected		| inputs 120, outputs 84						|
| RELU					|												|
| Fully connected		| inputs 84, outputs 43							|
| Softmax				| 	        									|
