
# # **Traffic Sign Recognition**

# Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer. 

**Build a Traffic Sign Recognition Project**
# The goals / steps of this project are the following: 
* Load the data set (see below for links to the project data set) 
* Explore, summarize and visualize the data set 
* Design, train and test a model architecture 
* Use the model to make predictions on new images 
* Analyze the softmax probabilities of the new images 
* Summarize the results with a written report 
# [//]: # (Image References) 

[image1]: ./examples/visualization.jpg "Visualization" [image2]: ./examples/grayscale.jpg "Grayscaling" [image3]: ./examples/random_noise.jpg "Random Noise" [image4]: ./examples/placeholder.png "Traffic Sign 1" [image5]: ./examples/placeholder.png "Traffic Sign 2" [image6]: ./examples/placeholder.png "Traffic Sign 3" [image7]: ./examples/placeholder.png "Traffic Sign 4" [image8]: ./examples/placeholder.png "Traffic Sign 5"

[image1]: ./examples/visualization.jpg "Visualization" [image2]: ./examples/grayscale.jpg "Grayscaling" [image3]: ./examples/random_noise.jpg "Random Noise" [image4]: ./examples/placeholder.png "Traffic Sign 1" [image5]: ./examples/placeholder.png "Traffic Sign 2" [image6]: ./examples/placeholder.png "Traffic Sign 3" [image7]: ./examples/placeholder.png "Traffic Sign 4" [image8]: ./examples/placeholder.png "Traffic Sign 5"

[image1]: ./examples/visualization.jpg "Visualization" 
[image2]: ./examples/grayscale.jpg "Grayscaling" 
[image3]: ./examples/random_noise.jpg "Random Noise" 
[image4]: ./examples/placeholder.png "Traffic Sign 1" 
[image5]: ./examples/placeholder.png "Traffic Sign 2" 
[image6]: ./examples/placeholder.png "Traffic Sign 3" 
[image7]: ./examples/placeholder.png "Traffic Sign 4" 
[image8]: ./examples/placeholder.png "Traffic Sign 5"

# ## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation

# ### Writeup / README 

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.    

# Data Set Summary & Exploration # ### Data Set Summary & Exploration 

#  Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set: 
 
Number of training examples = 34799


Number of testing examples = 12630

Image data shape = (32, 32, 3)

Number of classes = 43

# Include an exploratory visualization of the dataset.

Here I have plotted an exploratory visualization of the dataset. Shown below is the bar chart visualization of validation,training abd test data:

![image.png](attachment:image.png)

![image.png](attachment:image.png)

![image.png](attachment:image.png)

# Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the image to a grayscale image because it reduces time taken in processing as the image is monotonous rather than having 3 layers which is more time consuming and even it takes longer to make predictions.
The grayscale pixel intensities are unsigned integer values, with the values of the pixels falling in the range (0,255).

Then, I normalized the image data because wider distribution of data makes it more difficult to train by single learning rate.So, after normalization, with different features having different ranges weight can be diverged by single learning rate.

Original image set:

![image.png](attachment:image.png)

Output after preprocessing:

![image.png](attachment:image.png)

#  Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model layer description :

Input : 32x32x1 grayscale image

Convolution 5x5 :  2x2 stride, valid padding, outputs 28x28x6 

RELU 

Max pooling     :  2x2 stride, outputs 14x14x6 

Convolution 5x5 :  2x2 stride, valid padding, outputs 10x10x16  

RELU 

Max pooling     :  2x2 stride, outputs 5x5x16 

Convolution 1x1 :  2x2 stride, valid padding, outputs 1x1x412 

RELU 

Fully connected :  input 412, output 122

RELU
    
Dropout

Fully connected :  input 122, output 84 

RELU 

Dropout

Fully connected :  input 84, output 43 

# Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

While training the LeNet Architecture I modified the following values for the hyperparameters:

Epochs = 20

Batch_size = 138

Learning rate = 0.00097

I changed the epochs, batch size and learning rate. Before deciding to keep it this way and on further modifications i achieved test accuracy of around 94.4

#  Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem. 

In my final model output I was able to achieve:

• training set accuracy of 99.9% 

• validation set accuracy of 95.5% 

• test set accuracy of 94.4%
    

If an iterative approach was chosen:

• What was the first architecture that was tried and why was it chosen?

LeNet architecture was used,it would be similar architecture offered to us. Itwas easy to understand and has given me a good score.

• What were some problems with the initial architecture?

Lack of data and lack of knowledge of other parameters. Adding few convolution layers and further playing with the hyperparameters helped me reaching higher accuracy.

• How was the architecture adjusted and why was it adjusted? 

It was adjusted by adding a few convolution layers and a couple dropouts to the previous architecture as it was not able to achieve the desired accuracy.

• Which parameters were tuned? How were they adjusted and why?

o Epoch: I played around with this and set it to 20 as I was able to achieve good value for accuracy. 
o I increased the batch size as learned we can increase the data size if memory allows it . 
o The learning rate I think could have been left at .001 as it was default good rate by I tried changing it as it mattered a little  

• What are some of the important design choices and why were they chosen?

The original LeNet architecture used TANH,the reason we use RELU as it gives better classification accuracy due to a number desirable properties Convolution preserves the spatial relationship between pixels by learning image features using small filter square over the input data.These filters acts as feature detectors from the original input image.The more number of filters we have, the more image features get extracted and the better our network becomes at recognizing patterns in unseen images.

It provide me with such learning opportunity that I could take weeks more to learn. I could still learn more about that. I think the  thing I learned was having a more uniform dataset along with enough convolutions to capture features will improve speed of training and accuracy.

# Test a Model on New Images

Following are the 10 German traffic signs that I reffered on web
![image.png](attachment:image.png)

# Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model correctly guessed 8 of the 10 traffic signs,hence giving an accuracy of 80%.

# Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

shown below is the image of output predicted for the web images:

![image.png](attachment:image.png)

The forth and the fifth images were not identified correctly as the image for an right-of-way at the next intersection 
looks similar to that of Road work and similarly for the other, when an image of 32*32 is used and the image gets low in 
resolution when converted to gray scale and normalization is applied thus it almost looks similar to 
output class predicted by the model.
