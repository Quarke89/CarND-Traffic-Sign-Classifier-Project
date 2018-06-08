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

[image11]: ./writeup/images/performance_default.jpg "performance_default"
[image12]: ./writeup/images/performance_norm.jpg "performance_norm"
[image13]: ./writeup/images/performance_norm_gray.jpg "performance_norm_gray"
[image14]: ./writeup/images/performance_norm_gray_dropout.jpg "performance_norm_gray_dropout"
[image15]: ./writeup/images/performance_norm_gray_dropout_aug.jpg "performance_norm_gray_dropout_aug"

[image16]: ./test_images/1.jpg "test1"
[image17]: ./test_images/2.jpg "test2"
[image18]: ./test_images/3.jpg "test3"
[image19]: ./test_images/4.jpg "test4"
[image20]: ./test_images/5.jpg "test5"

[image21]: ./writeup/images/softmax_image1.jpg "softmax_image1"
[image22]: ./writeup/images/softmax_image2.jpg "softmax_image2"
[image23]: ./writeup/images/softmax_image3.jpg "softmax_image3"
[image24]: ./writeup/images/softmax_image4.jpg "softmax_image4"
[image25]: ./writeup/images/softmax_image5.jpg "softmax_image5"

[image26]: ./writeup/images/visualize_network.JPG "visualize_network"


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

The final model is the LeNet architecture with dropout and consisted of the following layers

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
| Dropout				| 50%											|
| RELU					|												|
| Fully connected		| inputs 120, outputs 84						|
| Dropout				| 50%											|
| RELU					|												|
| Fully connected		| inputs 84, outputs 43							|
| Softmax				| 	        									|

The model used the following values for the hyper-parameters

* Epochs: 50
* Batch size: 256
* Learning rate: 0.001
* Dropout: 50%

Using the above pre-processing techniques and hyper-parameters, the model was above the achieve the following performance:

* Training Set: 99.7%
* Validation Set: 99.0%
* Test Set: 94.0%

The LeNet architecture is a proven architecture in the literature for image classification and was a good starting point for this project. Dropout layers were added after the fully connected layers in order to reduce overfitting on the training set and the output layer was of course modified to compute 43 classes instead of 10. 
Most of the optimization was done on the type of pre-processing used and the hyper-parameters to achieve the performance listed above. Optimization of the parameters was done by sweeping over multiple values for the hyper-parameters and measuring the performance on the validation set. The Adam optimizer was used to minimize the cross entropy loss as its a faster and more optimized version of gradient descent

The following shows the performance of the model on the training and validation set over 25 epochs and the evolution of the optimization. Once the hyper-parameters and pre-processing was finalized I increased the epochs to 50 before checking the performance on the test set


**Initial performance** (Train: 99.2%, Validation: 87% after 25 epochs) 

![alt test][image11]                          
This is the default performance of the model with learning rate of 0.001 and batch size of 256 with no pre-processing applied to the input. Convergence is slow and there is a big gap between the training accuracy and the validation accuracy suggesting an overfitting problem			

**Normalizing input** (Train: 99.6%, Validation: 91.1% after 25 epochs) 

![alt test][image12]    

Normalizing the input images increased the training and validation accuracy in the inital few epochs as it helps speed up the training convergence and also helped the final validation accuracy by 4% after 25 epochs

**Grayscale conversion** (Train: 99.7%, Validation: 90.7% after 25 epochs) 

![alt test][image13]                         

Grayscale convergence did not have a big impact on the model performance but it helps speed up the computation as it reduces the number of calculations required 

**Adding dropout** (Train: 99.2%, Validation: 95% after 25 epochs) 

![alt test][image14]              

Adding dropout helped reduce the gap between the training and validation set as it reduces model overfitting. A dropout of 50% gave the best improvement after checking over different dropout percentages. A 95% validation accuracy was achieved, above the target goal           

**Augmenting data set** (Train: 99.4%, Validation: 98.8% after 25 epochs) 

![alt test][image15]                    

Augmenting the data set as described in the pre-processing section had a big impact on the model performance as the input images were distributed more evenly among the classes. A validation accuracy of 98.8% was achieved      


### Test a Model on New Images

#### Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. Shown below are the original images. They had to be resized, normalized, and converted to grayscale before they could be input into the network

Since the original images are much bigger than the required input size, there will be some loss in resolution that could potentially make the signs harder to classify. Image 3 is a speed limit sign that is slightly rotated so the network could potetially confuse it with other speed limit signs. Image 4 also a few signs in the background

![alt text][image16] ![alt text][image17] ![alt text][image18] 
![alt text][image19] ![alt text][image20]

After running these images through the network, it correctly identified all 5 of them (100% accuracy). The softmax probabilities for each image were also very high on the correct class index which suggests the images chosen were actually fairly easy for the network to identify


Here are the results of the prediction:

| Image			        						|     Prediction	        					| 
|:---------------------------------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection    		| Right-of-way at the next intersection  		| 
| Yield    										| Yield 										|
| Speed limit (70km/h)							| Speed limit (70km/h)							|
| Turn right ahead:	      						| Turn right ahead:					 			|
| No entry										| No entry      								|

**Softmax probabilities and top 5 classes**

**Image 1**

![alt text][image21]

* Right-of-way at the next intersection: 1.000
* Beware of ice/snow: 0.000
* Priority road: 0.000
* End of no passing: 0.000
* Pedestrians: 0.000

**Image 2**

![alt text][image22]

* Yield: 1.000
* No passing: 0.000
* No passing for vehicles over 3.5 metric tons: 0.000
* No vehicles: 0.000
* Keep right: 0.000

**Image 3**

![alt text][image23]

* Speed limit (70km/h): 0.999
* Speed limit (20km/h): 0.000
* Speed limit (120km/h): 0.000
* Speed limit (30km/h): 0.000
* Speed limit (100km/h): 0.000

Although the network was certain that the sign was 70km/h its interesting to note that the next 4 choices were also speed limit signs

**Image 4**

![alt text][image24]

* Turn right ahead: 1.000
* Ahead only: 0.000
* Speed limit (30km/h): 0.000
* Yield: 0.000
* No vehicles: 0.000

**Image 5**

![alt text][image25]

* No entry: 1.000
* Priority road: 0.000
* Yield: 0.000
* No passing: 0.000
* End of no passing: 0.000

### Visualizing the Neural Network 

The figure below shows the feature maps of the first layer on the 70km/h sign. You can see how the filter weights are trying to extract the shape of the sign as well as the letters inside

![alt text][image26]