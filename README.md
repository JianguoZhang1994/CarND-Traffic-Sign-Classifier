## Project: Build a Traffic Sign Recognition Program
**Jianguo Zhang, March 29, 2017**

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We have included an Ipython notebook that contains further instructions 
and starter code. Be sure to download the [Ipython notebook](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). 

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]:  Number_of_each_sign.png
[image2]:  5_images_or.png
[image3]:  5_images_re.png
[image4]:  straight_or_right.jpg
[image5]:  ahead_only.jpg
[image6]:  wild_animal_crossing.jpg
[image7]:  stop.jpg

---
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


Here is an exploratory visualization of the data set. It is a chart showing the number of each sign in training dataset

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.


As a first step, I normalize images for training dataset, validation dataset and test dataset, the scale is 0-1. Where I did not grayscale the image and keep 3 channels for each image. 

 

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
| Fully connected		| Input  400, Output  120      			    	|
| RELU					|												|
| DROP					| Keep Prob=0.5 for train, 1.0 for val and test	|
| Fully connected		| Inputs 120, Outputs 43      				    |

 


#### 3. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


To train the model, I used the AdamOptimizer method, more information in the training pipeline code cell

#### 4. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.



My final model results were:
* validation set accuracy of 0.940
* test set accuracy of 0.935

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? Oringinal LeNet 
* What were some problems with the initial architecture? Original LeNet includes 3 full connected layers, too easy overfitting
* How was the architecture adjusted and why was it adjusted? (Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.) I modify the LeNet model, reduce the full connected layers to 2 layers, and add dropout layer after the first full connected layer. 
* Which parameters were tuned? How were they adjusted and why? I finetuning parameters of epoches, batch_size and learning rate. The accuracy will stop learn after certain epoches, too many epoches will waste time, I set epoches to 15. The batch_size is 128, too smaller batch size will take up huge amount of memory, too large maybe affect accuracy. Learning rate is 0.001  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? Drouput set to 0.5, using dropout will improve accurace significantly.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2]  

Due to original images are not suitable for the model, I resize each image to 32x32x3.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


|ClassId       | Image			        |     Prediction	        					| 
|:------------:|:---------------------:|:---------------------------------------------:| 
|25            | Road work   		| Road work								| 
|23            | Slippery road    		| Slippery road						| 
|36            | Go straight or right  | Go straight or right				|
|35            | Ahead only			|  Ahead only            				|
|31            | Wild animals crossing | Wild animals crossing 				|
|14            | Stop		| Stop      |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.935.

However, when I test the model on a new speed limit sign images, it is hard to recognize correct prediction, for other traffic signs it can successfully predict. Besides, images from Google street view maps can work, but lots of internet images cannot be recognized by the model, I don't know why.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

Take the fouth slippery Ahead only sign image as an example, the model is relatively sure that this is a Ahead only sign, and the image does contain a Ahead only sign. The top five soft max CarId and predictions were
35 36 33 39 38

| ClassId           	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 35         			| Ahead only  								| 
| 36     				| Go straight or right									|
| 33					| Turn right ahead										|
| 39	      			| Keep left	|
| 38 				    | Keep right							|

 

