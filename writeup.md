# **Traffic Sign Recognition** 

## Writeup 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histogram.PNG "Visualization"
[image2]: ./examples/sample_data.PNG "Sample data"
[image3]: ./examples/preprocessed.PNG "Transformation"
[image31]: ./examples/augmented.PNG "Augmented"
[image4]: ./examples/googlenet_diagram.png "Traffic Sign 1"
[image5]: ./examples/new-images.png "Traffic Sign 2"
[image6]: ./examples/prediction-new-images.png "Traffic Sign 3"
[image7]: ./examples/top_3.png "probability"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


Here is a link to my [project code](https://github.com/ledrui/Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. 

In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used pickle to load the training, validation, and testing data.
I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. An exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution by class.

![Sample data][image1]
![Histogram][image2]
After plotting the distribution of the number of images for each class, it shows how imbalance the dataset was.  An imbalanced dataset can lead the model to be biased toward classes that are highly represented. At first, I was inclined to balance the dataset. However, this dataset is the representation of the prior probability of occurrence of each traffic sign in real life. Keeping the relative size of each class will help the model, in case of confusion, predict the class with the highest probability occurrence. 

### Design and Test a Model Architecture

#### 1. Preprocessing of the image data.

The traffic signs on raw images do not cover images from edge to edge. So, cropping was applied (resulted in image resolution reduction from 32x32 px to 26x26 px). It is also beneficial for network training speed because. We applied sharpen with unsharp mask, contrast enhance and histogram equalization because the images shows lack of sharpen, brightness and contrast. All operations are called in the transform_img function.
We saved colors(no grayscale conversion) because color can be the key to traffic sign classification (for example, sign background and edging colors). In contrary to the Pierre Sermanet and Yann LeCun results, after some experiments conducted on the CNN architecture with grayscale input images has shown worse results..

Here is an example of a traffic sign image before and after transformation.

![alt text][image3]

For a deep neural network to generalize well need to train it on enough data (there is more to it, but ...). Here we have limited data and we don't have the convenience of collecting more data, however, we can augment the dataset by applying some data processing techniques like affine transformation to jitter the images to produce more images. An affine transformation is a transformation that keeps the parallelism of the line.

To add more data to the the data set, I used the following techniques because:
* Increased contrast
* Scalling
* Rotating

Here is an example of an original image and an augmented image:

![alt text][image31]
 


#### 2. Final model architecture

![alt text][image4]

My final model consisted of the following layers

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 26x26x3 RGB image   							| 
| Convolution1       	| 1x1 stride, same padding, outputs 26x26x3 	|
| RELU                  |                                               |
| Convolution2       	| 5x5 stride, same padding, outputs 26x26x32 	|
| RELU                  | 	    										|
| INCEPTION_1     	    |                          outputs 13x13x256 	|
| Max pooling	      	| 2x2 stride,              outputs 13x13x256	|
| INCEPTION_2     	    |                          outputs 13x13x512 	|
| Max pooling	      	| 3x3 stride,              outputs 6x6x512  	|
| Convolution3    	    | 1x1 stride, same padding, outputs 6x6x256     |
| Fully connected1		| outputs 512        							|
| Fully connected2		| outputs 512        							|
| Softmax				|           									|


 


#### 3. Describe how you trained your model.

The model was trained with the Adam optimizer, batch size = 128 images, initial learning rate was 0.0005 with decrease by 20% every 5 epochs. The model was trained for 20 epochs (102400 images in each epoch) with one dataset. The model stopped improving after 10 epochs.

Variables were initialized with tensorflow using of a truncated normal distribution with mu = 0.0 and sigma = 0.1. Learning rate was finetuned by try and error process.

Traffic sign classes were coded into one-hot encodings.

Detailed training log can be found in the train_log_f2.csv and train_log_f.csv files (it includes minibatch loss and accuracy)
Training was performed on a custom built GTX 1070 GPU and it takes about one hour.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

I used the well known google GoogLeNet architecture for this project. It was a trial and error process

* The model is based on google GoogLeNet. It is a convolutional neural network designed to recognize visual patterns directly from pixel images with minimal preprocessing. It can handle more than 1000 classes, was the winner of 2014 ImageNet LSVRC challenge. It's a proven model for image classification.

My final model results were:
* training set accuracy of 95%
* validation set accuracy of 95% 
* test set accuracy of 94% - pretty good generalization

As it is an image classification problem, convolutional layers were used as it is common in modern image classification systems as reduce computation (as compared to classical neural network with only fc layers). It seems to me that inception moduls are essential for good performance on such kind of tasks as they allow to do not select optimal layer (say, convolution 5x5 or 3x3), by perform different layer types simultaneously and it selects the best one on its own.



#### Future Improvement
* I want to try the following to see if I can improve the performance while not causing any overfit:
* more filters in conv layers
* more neurons in dense layers
* more conv layers
* more dense layers
* different activation like elu instead of relu
* dropout
* The GoogLeNet model is a bit of overkill for this project, I was thinking experiment with lesser complex model like LeNet5


### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 12 German traffic signs that I found on the web:

![alt text][image5] 

There are 10 real traffic signs from real world streets corresponding to the presented images in the train dataset classes. They have a little bit different to the test dataset colors and arrows style. Some traffic signs are dirty or images are light damaged.

There are 2 additional road signs missing in the training dataset ( "No U-turn" and "speed limit (40 km / h)"). They should not be used to assess the accuracy, but included as experiment.


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image6] 

We can see that 2 of traffic signs of presented in the training set classes were misclassified ("speed limit (30 km / h)" and "speed limit (50 km / h)"). In both cases net correctly respond that they were a speed limit signs, so only numbers were recognized incorrectly. (One of the possible solution for real world application is to train another NN to classify numbers only). So, accuracy is about 80% on the real world extra images.

Extra sign "speed limit (40 km / h)" was misclassified (obviously) as a another speed limit sign, so, the model can understand traffic sign types quite well. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 36th cell of the Ipython notebook.



![alt text][image7]

In all cases except the extra sign "No U-turn" the predictor was very certain. In case of misclassified speed limit signs the correct predictions were the second probable option. "No vehicles" presented some difficulties as it has a big hotspot, but was classified correctly. So, the system is quite reliable and with some modifications it could perform really well on real data.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


