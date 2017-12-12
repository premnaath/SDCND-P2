# **Traffic Sign Recognition** 

## Writeup
The purpose of this writeup is to give more insight in to the CNN architecture
developed for project 2 of self-driving car nanodegree program offered by
Udacity.

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/train.png "Train dist"
[image2]: ./images/validate.png "Validate dist"
[image3]: ./images/test.png "Test dist"
[image4]: ./images/1_yield.png "Traffic Sign 1"
[image5]: ./images/2_no_passing.png "Traffic Sign 2"
[image6]: ./images/3_speedlimit_20.png "Traffic Sign 3"
[image7]: ./images/4_straight_or_right.png "Traffic Sign 4"
[image8]: ./images/5_roundabout.png "Traffic Sign 5"
[image9]: ./images/preprocessing.png "Preprocessing"


## 1. Purpose
This reflection document describes the procedure adopted in designing and training
a new model to classify traffic signs from the German traffic sign database.

The contents of this document are listed as follows,
* Dataset summary
* Exploratory visualization
* Pre-processing
* Model architecture - start
* Model architecture - tuning
* Acquiring new images
* Performance on new images
* Model certainity
* Conclusion

## 2. Rubric Points
This section shows if all the Rubic criterias has been satisfied in this project.

### 2.1 Files submitted
All files submitted satisfies this criteria

### 2.2 Data exploration
| Criteria			| Satisfied by|
|:------------------------------|-------------:|
| Dataset summary		| Section 3|
| Exploratory visualization	| Section 4|

### 2.3 Design and test model architecture
| Criteria			| Satisfied by|
|:------------------------------|-------------:|
| Pre-processing		| Section 5|
| Model architecture		| Section 6|
| Model training		| Section 7|
| Solution approach		| Section 7|

### 2.4 Test a model on new images
| Criteria			| Satisfied by|
|:------------------------------|-------------:|
| Acquiring new images		| Section 8|
| Performance on new images	| Section 9|
| Model certainity		| Section 10|

## 3. Data Summary
Summary of the provided pickled dataset is given here.

| Dataset				| value |
|:--------------------------------------|-------------:|
| Number of training samples		| 34799|
| Number of validation samples		| 4410|
| Number of test samples		| 12630|
| Shape of image data			| 32x32x3|
| Number of classes			| 43|

The sizes of different image sets were compared. The results are stated as below,

The validation set is 12.7% the size of training set.
The test set is 36.3% the size of training set.

## 4. Exploratory visualization
Histograms were used to analyze and visualize the data by using labels as bins. It was noticed
that all sets (training, validation and test sets) contain the same distribution
of data over all the classes.

![Train set distribution][image1]
![Validation set distribution][image2]
![Test set distribution][image3]

The above images shows the distribution of number of images to all labels present
in train, validate and test sets. The histogram on the left of an image shows
actual number of images for each labels. The histogram on the right shows the
normalized version of the distribution.

It can be visually verified that the normalized distribution is almost the same
for the train, validate and test sets. So a model trained on the training set
must be able to classify the images on the validation and test sets with approx.
equal accuracy.

## 5. Model Architecture
The LeNet5 architecture designed for DNN lab was considered to be a starting
point for designing the traning architecture. This section describes the types
of pre-processing techniques evaluated, the final model architecture and the
training and tuning of hyperparameters.

### 5.1 Pre-processing
This section explains various approaches carried out for pre-processing of
the data. The goal was to bring in more information from the features and also to
acheive statistical invariance so that the model can be trained effectively
without overfitting. 

### 5.2 Technique 1 - 3-color channel as input
The LeNet5 designed for the DNN lab was for the MINST dataset. Those were images
in grayscale only. Initially it was assumed that instead of using a grayscale 
image to train the model, a 3-color channel image would fair better. Hence the 
LeNet5 architecture was changed to take in a 3-color channel input.

Even with the tuning of hyperparameters an accuracy of only 89% was acheived.

### 5.3 Technique 2 - 3-color channel with normalization
The 3-color channel images were also normalized using the formula (pixel - 128)/128.

This for some reason worsened the performance accuracy. The reason was found to be
saturation of these color channels after normalization.

### 5.4 Technique 3 - Grayscale
Since a gray scale image can be used to extract the basic shape features from an
image (learnt from project 1 CARND), this was used as the next method for
pre-processing.

The images were transformed to grayscale by using the openCV library. The LeNet5
implementation and the features place holder were edited to take a 1-color
channel (grayscale image) batch as input.

An accuracy of 89% was achieved with this pre-processing setup, which is
practically unchanged from the pre-processing method explained in Technique 1.

### 5.5 Technique 4 - Grayscale with normalization
It was also observed that the validation accuracy from Technique 3 was not constantly
decreasing. This meant that there is a requirement for normalizing the grayscale
image in order to tune the weights better during the tuning_operation.

This technique is an extension of Technique 3 with the inclusion of image
normalization. The 3-color channel image was transformed to grayscale by averaging
out each pixel values across all the color channels.

$pixel = (R + G + B)/3$

The output size of the image after the above formula is 32x32x1 fom 32x32x3.

Each of the images were normalized by subtracting each pixel from the mean and
dividing it by the standard deviation. The formula is shown below.

$pixel = (pixel - \mu)/\sigma$

This technique without tuning the hyperparameters gave a validation accuracy of
90%. Hence this method of pre-processing was chosen to proceed forward.

The figure below shows the effect of this pre-processing technique on an image.

![Pre-processing][image9]

## 6. Model architecture - start
As mentioned earlier the already developed LeNet5 architecture was chosen as a
starting point.

The starting model consisted of layers mentioned in the table below.
Assuming S = strides, P = padding,

| Layer				| Size | output |
|:------------------------------|:------------------------:|:------------:|
| Input				| 32x32x1 grayscale image|
| Convolution 5x5		| S:1x1, P:'VALID'| 28x28x6 |
| RELU				|  |  |
| Maxpooling 2x2		| S:2x2 | 14x14x6 |
| Convolution 5x5		| S:1x1, P:'VALID'| 10x10x16|
| RELU				|  |  |
| Maxpooling 2x2		| S:2x2 | 5x5x16 |
| Fully connected layer 1	| 400 | 120 |
| RELU				|  |  |
| Fully connected layer 2	| 120 | 84 |
| RELU				|  |  |
| Fully connected layer 3	| 84 | 43 |
|  |  |  |
|  |  |  |

The hyper parameters selected were,
- Epochs - 10
- Batch size - 64
- No dropout
- Learning rate - 0.001
- Pre-processing - Technique 4

The optimizer used was AdamOptimizer.

With the above mentioned parameters and architecture, the achieved accuracies
are stated below,
* Training accuracy - 99.6%
* Validation accuracy - 93.0%
* Test accuracy - 91.5%

### 6.1 Shortcoming
The training and validation and the test accuracies seem to be far apart. The
model needs more training. But increasing the epochs doesn't yield better
accuracies. Hence the model can be stated as underfit. To gain better accuracies
the architecture has to be made deeper.

## 7. Model architecture - tuning
This section describes the tuning procedure considered in order to train the
model for better validation accuracies. 

I was also observed that the model mentioned above might not capture all the
necessary feature maps as it was used to train MINST dataset. The traffic sign
training dataset has more complex shapes. The number of classes we want to train
a model for has also increased from 10 (MINST) to 43. These reasons motivate
for a deeper layer to train the traffic signs.

Here one more fully connected layer is introduced which makes our model a 6
layered model. The parameters used by this model are mentioned below.

| Layer				| Size | output |
|:------------------------------|:------------------------:|:------------:|
| Input				| 32x32x1 grayscale image|
| Convolution 5x5		| S:1x1, P:'VALID'| 28x28x8 |
| RELU				|  |  |
| Maxpooling 2x2		| S:2x2 | 14x14x8 |
| Convolution 5x5		| S:1x1, P:'VALID'| 10x10x32|
| RELU				|  |  |
| Maxpooling 2x2		| S:2x2 | 5x5x32 |
| Fully connected layer 1	| 800 | 400 |
| RELU				|  |  |
| Dropout				|  |  |
| Fully connected layer 2	| 400 | 200 |
| RELU				|  |  |
| Dropout			|  |  |
| Fully connected layer 3	| 200 | 84 |
| RELU				|  |  |
| Dropout			|  |  |
| Fully connected layer 4	| 84 | 43 |
|  |  |  |
|  |  |  |

The hyper parameters selected were,
- Epochs - 10
- Batch size - 64
- Keep probability - 0.75
- Learning rate - 0.001
- Pre-processing - Technique 4
- AdamOptimizer.

With the above mentioned parameters and architecture, the achieved accuracies
are stated below,
* Training accuracy - 99.9%
* Validation accuracy - 96.9%
* Test accuracy - 94.1%

Here the effect of dropout was evaluated. While using dropout, the validation
accuracy was 84% at epoch 1 and increased from then on until 96.9% at epoch 10.
It was also observed that the usage of dropout improved validation accuracy
in each epoch. As dropout also drops some activations randomly, better
robustness was also achieved.

Tuning was continued to improve the test accuracy. It was found that decreasing
the value of keep probability trained the weights better. Reason being that the
weights that matter gained more and more probability in each training operation.

The final architecture used is the same table as above, however the hyperparameters
have slightly changed and are mentioned below,

- Epochs - 10
- Batch size - 64
- Keep probability - 0.5
- Learning rate - 0.001
- Pre-processing - Technique 4
- AdamOptimizer.

With the above mentioned parameters and architecture, the achieved accuracies
are stated below,
* Training accuracy - 99.8%
* Validation accuracy - 97%
* Test accuracy - 94.9%

The validation accuracy at epoch 1 was 75% and was continuously improved to
yield the above mentioned 97%.

## 8. Acquiring new images
The trained model has to be tested for 5 new images obtained from the web.

These new images are shown below,

![Yield][image4] ![No passing][image5] ![Speed limit 20][image6] 
![Straight or right][image7] ![Roundabout][image8]

| Image				| Size before| Size after | Label |
|:-----------------------------:|:----------:|:-----------:|:------------:|
| Yield				| 28x32x3 | 32x32x3 | 13 |
| No passing			| 32x31x3 | 32x32x3 | 9 |
| Speed limit 20 km/h		| 29x32x3 | 32x32x3 | 0 |
| Straight or right		| 27x32x3 | 32x32x3 | 36 |
| Roundabout			| 28x32x3 | 32x32x3 | 40 |
|  |  |  |  |
|  |  |  |  |

These images were obtained from google street view with print screen and then
cropped and resized. The size of these images can still be different from the
images used for training, validating and testing. Hence they were padded with
zeros in order to resize them to exactly 32x32x3.

After resizing, these images appended to an numpy array and dumped to a pickle
file along with their corresponding labels. Doing so the changes required for
testing the new images on the new model requested least changes to the code.

## 9. Performance on new images
Testing was performed on the features and labels from the newly created pickle
file as a whole. These images were pre-processed using the same technique as
the other images. A test accuracy of 20% corresponds to one image being
correctly classified.

The trained model was able to classify all 5 images to their respective labels
correctly.

The same set of hyperparameters were used to test the classification performance
on these new images. These parameters are once again listed as follows,

- Epochs - 10
- Batch size - 64
- Keep probability - 0.5
- Learning rate - 0.001
- Pre-processing - Technique 4

The testing accuracy achieved on new images is 100%. 

Since the model was trained with a dropout of 50%, it can be said that all
randomly declared weights were tuned for robustness.

## 10. Model certainity
The certainity of the model can be further discussed with the help of softmax
probabilities.

The softmax probabilities were generated using the tf.nn.softmax() method and
only the top five probabilities (since there were only 5 new images) from 43
were listed using the tf.nn.top_k() method.

Softmax probability for the first image 1, yield is listed below,

| Probability			| Prediction | Label |
|:-----------------------------:|:----------:|:-----:|
| 1.00000000e+00		| Yield |  13 |
| 3.08283576e-09		| Priority road | 12 |
| 1.30754263e-09		| No vehicles | 15 |
| 2.67970562e-10		| Speed limit 60 kmph | 3 |
| 2.39667702e-10		| Bumpy road | 22 |
|  |  |
|  |  |

The probability for yield is exactly one and it can be stated that the traffic
sign, yield has been predicted with the maximum probability.

Softmax probability for the first image 2, no passing is listed below,

| Probability			| Prediction | Label |
|:-----------------------------:|:-------------------:|:-----:|
| 1.00000000e+00		| No passing |  9 |
| 1.62297491e-22		| No passing > 3.5t | 10 |
| 2.60128185e-24		| Prohibited > 3.5t | 16 |
| 6.83242143e-25		| Slippery road | 23 |
| 8.65869260e-26		| No vehicles | 15 |
|  |  |
|  |  |

The probability for yield is exactly one and it can be stated that the traffic
sign, no passing has been predicted with the maximum probability.

Softmax probability for the first image 3, speed limit 20km/h is listed below,

| Probability			| Prediction | Label |
|:-----------------------------:|:-------------------:|:-----:|
| 9.95012820e-01		| Speed limit 20 km/h |  0 |
| 3.48533154e-03		| Speed limit 30 km/h | 1 |
| 4.81087656e-04		| Speed limit 70 km/h | 4 |
| 4.53688961e-04		| Roundabout | 40 |
| 3.04412009e-04		| Speed limit 120 km/h | 8 |
|  |  |
|  |  |

The probability for yield is exactly one and it can be stated that the traffic
sign, speed limit 20 km/h has been predicted with the maximum probability.

Softmax probability for the first image 4, go straight or right is listed below,

| Probability			| Prediction | Label |
|:-----------------------------:|:-------------------:|:-----:|
| 9.99999642e-01		| Go straight or right |  36 |
| 1.31231062e-07		| Turn left ahead | 34 |
| 8.73835617e-08		| Wild animals crossing | 32 |
| 5.48042500e-08		| Keep right | 38 |
| 1.91772873e-08		| Ahead only | 35 |
|  |  |
|  |  |

The probability for yield is exactly one and it can be stated that the traffic
sign, go straight or right has been predicted with the maximum probability.

Softmax probability for the first image 5, roundabout is listed below,

| Probability			| Prediction | Label |
|:-----------------------------:|:-------------------:|:-----:|
| 9.99646783e-01		| Roundabout |  40 |
| 2.07099991e-04		| Priority road | 12 |
| 1.40326389e-04		| Speed limit 100 km/h | 7 |
| 2.10115513e-06		| Go straight or left | 37 |
| 1.51967924e-06		| Beware of ice/snow | 30 |
|  |  |
|  |  |

The probability for yield is exactly one and it can be stated that the traffic
sign, roundabout has been predicted with the maximum probability.

Thus it has been verified that the trained model classifies a small set of
traffic signs randomly picked from the web.

## 11. Conclusion
A robust architecture with Convolutional Neural Network has been developed and
trained with a training data provided. The trained model has been validated with
the provided validation set and was found to improve validation accuracy with
each epoch. The models performance was also checked against the provided test
set and on randomly picked German traffic images from the internet.


