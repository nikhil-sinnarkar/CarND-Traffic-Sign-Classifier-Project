# **Traffic Sign Recognition** 


### The project analyzes the input image of german traffic sign and predicts the label for the image.

---

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


---
### Writeup

You're reading it! and here is a link to my [project code](https://github.com/nikhil-sinnarkar/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Here is the basic summary of the data set.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing training, validation and test data.
![bar_train](./writeup_images/training%20data%20graph.JPG)
![bar_valid](./writeup_images/validation%20data%20graph.JPG)
![bar_test](./writeup_images/test%20data%20graph.JPG)


### Design and Test a Model Architecture

#### 1. Preprocessing the image data.

Initially I trained my model on the given data set without any preprocessing and got around 86% accuracy. To increase the accuracy, I implemented preprocessing and got the accuracy up to 90%. This wasn't enough so to further increase it I went for data augmentation.

To augment the given training data I used rotation and brightness change. I increased the brightness of the images in training dataset by 20% and added them back to initial training data set.
```python
image = cv2.multiply(image_list[i], np.array([1.2]))
```
Then I ramdomly selected half of the images in training dataset and rotated them and added them to initial training dataset.
```python
rows,cols = image.shape[:2]
M = cv2.getRotationMatrix2D((cols/2,rows/2), 5, 1.2)
image = cv2.warpAffine(image,M,(cols,rows))
```
I converted this augmented training dataset into grayscale followed by normalization.
Here is an example of a traffic sign image before and after preprocessing.

![before1](./writeup_images/before1.JPG)
![after1](./writeup_images/after1.JPG)

![before2](./writeup_images/before2.JPG)
![after2](./writeup_images/after2.JPG)

![before3](./writeup_images/before3.JPG)
![after3](./writeup_images/after3.JPG)

![before4](./writeup_images/before4.JPG)
![after4](./writeup_images/after4.JPG)

![before5](./writeup_images/before5.JPG)
![after5](./writeup_images/after5.JPG)


#### 2. Final model architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image						| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	 				|
| Flatten				| outputs 400  									|
| Fully connected		| outputs 120									|
| RELU					|												|
| Dropout				| 												|
| Fully connected		| outputs 84									|
| RELU					|												|
| Dropout				| 												|
| Fully connected		| outputs 43									|
| Softmax				| 												|


#### 3. Training the model.

To train the model, I used Adam Optimizer with learning rate = 0.001

I changed the batch size to 100 and the number of epochs to 25.

#### 4. Getting the desired validation accuracy (i.e. > 0.93). 

I used the LeNet architecture for my CNN. First I ran the model as is and got the validation accuracy of 0.86, without any preprocrssing of training data. Next I normalized the images which increased the validation accuracy
to around 0.90.
I further increased the accuracy by doing data augmentation also I used dropout layer in between the fully connected layers in the LeNet architecture.

The final values of the parameters I used -
* No. of epochs = 25
* Batch size = 100
* Learning rate = 0.001
* Keep prob for dropout = 0.7

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.959 
* test set accuracy of 0.943
 

### Testing the Model on New Images

Here are seven German traffic signs that I found on the web:

![1](./webimages/1.jpg)	-	Stop

![2](./webimages/2.jpg)	-	Speed limit (70km/h)

![3](./webimages/3.jpg)	-	Roundabout mandatory

![4](./webimages/4.jpg)	-	No entry

![5](./webimages/5.jpg)	-	Turn left ahead

![6](./webimages/6.jpg)	-	Wild animals crossing

![7](./webimages/7.jpg)	-	Right-of-way at the next intersection

Some of the image might be difficult to classify. For eg. The second image as there is very little difference form other speed limit signs. The fourth image of "No Entry" has many similar features to "STOP" sign, "left turn/right turn" sign.

#### Model's predictions on these new traffic signs.

Here are the results of the prediction:

| Image			        				|     Prediction	        			| 
|:-------------------------------------:|:-------------------------------------:| 
| Stop sign      						| Stop sign   							| 
| 70 km/h     							| 70 km/h 								|
| Roundabout mandatory					| Roundabout mandatory					|
| No entry	      						| No Entry 						 		|
| Turn left ahead						| Turn left ahead      					|
| Wild animals crossing 				| Wild animals crossing					|
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|


The model was able to correctly guess 7 of the 7 traffic signs, which gives an accuracy of 100%.

#### Softmax probabilities for each prediction.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top 5 soft max probabilities were

| Probability         		|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| 1.0         				| Stop sign   									| 
| 5.808381611237636e-19		| Speed limit (30km/h)							|
| 1.6342845176636277e-19	| Road work										|
| 1.9626558014898303e-20	| Yield 						 				|
| 2.4977282773580867e-24   	| Turn left ahead      							|


For the second image ...

| Probability         		|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| 0.990625262260437			| Speed limit (70km/h)   						| 
| 0.009373641572892666		| Speed limit (30km/h)							|
| 1.0292650358678657e-06	| Speed limit (20km/h)							|
| 4.76350894865966e-16		| Stop 							 				|
| 1.2416025993046268e-16   	| Speed limit (50km/h) 							| 

For the third image ...

| Probability         		|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| 1.0						| Roundabout mandatory   						| 
| 1.2043975842687682e-12	| Right-of-way at the next intersection			|
| 6.449767952251655e-15		| Priority Road 								|
| 1.968727127325885e-15		| Speed limit (30km/h)			 				|
| 1.1764644951113503e-15   	| Children crossing								| 

For the fourth image ...

| Probability         		|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| 0.48899656534194946		| No entry 				   						| 
| 0.45303094387054443		| Turn left ahead								|
| 0.035171810537576675		| Stop 											|
| 0.018280770629644394		| Keep right					 				|
| 0.0036373217590153217   	| Yield 		 								| 

For the fifth image ...

| Probability         		|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| 1.0						| Turn left Ahead  		 						| 
| 6.099916765241005e-19		| Keep right									|
| 8.904519807246355e-20 	| Speed limit (30km/h)							|
| 1.8049882701022258e-21	| Roundabout mandatory			 				|
| 1.5343883303883637e-21   	| Ahead only		 							| 

For the sixth image ...

| Probability         		|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| 1.0						| Wild animals crossing   						| 
| 8.511694940275641e-23 	| Double curve									|
| 5.739182190576888e-33 	| Dangerous curve to the left					|
| 8.432326288916581e-37 	| Road work						 				|
| 6.424694498973079e-38   	| Bicycles crossing								| 

For the seventh image ...

| Probability         		|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| 1.0						| Right-of-way at the next intersection   		| 
| 1.7329427361630205e-28	| Beware of ice/snow							|
| 2.2679984243183293e-29	| Roundabout mandatory							|
| 1.5482513207109763e-30	| Pedestrians    					 			|
| 4.799873181235792e-35		| Priority Road 	 							| 

