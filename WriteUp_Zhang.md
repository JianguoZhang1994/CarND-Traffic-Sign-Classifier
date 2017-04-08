
# **Traffic Sign Recognition** 

## Writeup Template

### Jianguo Zhang

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]:  show_of_each_class.png
[image2]:  7_images_original.png
[image3]:  7_images_original.png
[image4]:  straight_or_right.jpg
[image5]:  ahead_only.jpg
[image6]:  wild_animal_crossing.jpg
[image7]:  stop.jpg
[image8]:  the_number_of_each_class.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/JianguoZhang1994/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.


Here is an exploratory visualization of each sign and its number in the data set. 

![alt text][image1]
![alt text][image8]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.


As a first step, I normalize images for training dataset, validation dataset and test dataset, the scale is 0-1. Where I did not grayscale the image and keep 3 channels for each image. 

Actually at the beggining, I set the scale as -1 to 1, but It makes my accuracy hard to beyonf 93%, I don't know why. Besides, grayscale can improve my accuracy by 2%, but when I deal with the new images downloaded from Internet, I know how to convert the image with size like 400x300x3 to 32x32, I really don't know how to get 32x32x1. So finally I did not use grayscale. 

 

#### 2. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I modify the LeNet model. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 2x2 ksize, outputs 14x14x6		|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, 2x2 ksize, outputs 5x5x16 		|
| Flatten   	      	| Inputs 5x5x16, outputs 400            		|
| Fully connected		| Input  400, Output  200      			    	|
| RELU					|												|
| DROP					| Keep Prob=0.5 for train, 1.0 for val and test	|
| Fully connected		| Inputs 200, Outputs 43      				    |

 


#### 3. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


To train the model, I used the AdamOptimizer method, the batch size is 128, epoches is 15, learnning rate is 0.001. 
 At the begginning, I want to define a max number of epochs, and on each epoch decide to continue or terminate based on the previous values for validation accuracy and/or loss. But I don't know how to write the codes.
 
Next time I want to try to use tensorboard to decide the best number of these parameters.

#### 4. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.



My final model results were:
* validation set accuracy of 0.940
* test set accuracy of 0.934

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? Oringinal LeNet 
* What were some problems with the initial architecture? Original LeNet includes 3 full connected layers, too easy overfitting
* How was the architecture adjusted and why was it adjusted? (Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.) I modify the LeNet model, reduce the full connected layers to 2 layers, and add dropout layer after the first full connected layer. 
* Which parameters were tuned? How were they adjusted and why? I finetuning parameters of epoches, batch_size and learning rate. The accuracy will stop learn after certain epoches, too many epoches will waste time, I set epoches to 15. The batch_size is 128, too smaller batch size will take up huge amount of memory, too large maybe affect accuracy. Learning rate is 0.001  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? Drouput set to 0.5, using dropout will improve accurace significantly.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image2]  

The original size are :

image 1 original shape:  (434, 468, 3)
image 2 original shape:  (412, 468, 3)
image 3 original shape:  (414, 413, 3)
image 4 original shape:  (396, 468, 3)
image 5 original shape:  (208, 235, 3)
image 6 original shape:  (319, 327, 3)
image 7 original shape:  (344, 353, 3)

Due to original images are not suitable for the model, I resize each image to 32x32x3. Some original images like 'Road work' image and 'Wild animals crossing' image have backgound which can affect classification. Besides, some images like 'Road work' after resized the core area of new image may become unclear. Which may cause the new image be miscalssified as some other similar signs like 'Vehicles over 3.5 metric tons prohibited'.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


|ClassId       | Image			        |     Prediction	        					| 
|:------------:|:---------------------:|:---------------------------------------------:| 
|25            | Road work   		| Road work								| 
|23            | Slippery road    		| Slippery road						| 
|36            | Turn left  | Roundabout mandatory				|
|35            | Ahead only			|  Ahead only            				|
|31            | Wild animals crossing | Wild animals crossing 				|
|14            | Stop		| Stop      |
|3             |Speed limit (60km/h)|Speed limit (60km/h)|

The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.7%. This compares less than the accuracy on the test set of 0.935. 'Turn left' image miscalssified as 'Roundabout mandatory' image. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


Take the second 'Slippery road' sign image as an example, the model is relatively sure that this is a Slippery road, and the image does belonged to the 'Slippery road' sign. The top five soft max CarId and predictions were
23 16 20  9 10

| ClassId           	|     Prediction	        					| Probability|
|:---------------------:|:---------------------------------------------:| ---------------:|
| 23         			| Slippery road  								| 9.53286111e-01|
| 16     				| Vehicles over 3.5 metric tons prohibited									| 4.55318950e-02 |
| 20					| Dangerous curve to the right										| 1.17826578e-03|
| 9	      			| No passing	|2.59197213e-06|
| 10 				    | No passing for vehicles over 3.5 metric tons							|8.58616659e-07|
      
   
 
