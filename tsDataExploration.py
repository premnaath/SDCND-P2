# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'
web_testing_file = 'web_img.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
with open(web_testing_file, mode='rb') as f:
    webtest = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
X_webtest, y_webtest = webtest['features'], webtest['labels']


########################################################################################################################
### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.max(y_train) + 1

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

########################################################################################################################
# Exploratory analysis of dataset

# Plotting a histogram for each data set.
num_bins = n_classes
'''
fig, axes = plt.subplots(nrows=1, ncols=3)
ax0, ax1, ax2 = axes.flatten()

ax0.hist(y_train, num_bins, normed=1)
ax1.hist(y_valid, num_bins, normed=1)
ax2.hist(y_test, num_bins, normed=1)

plt.xlabel('n bins')
plt.ylabel('class')
plt.show()
'''
fig_train, axes_train = plt.subplots(nrows=1, ncols=2)
ax0_train, ax1_train = axes_train.flatten()
n0_train, bins0_train, patches0_train = ax0_train.hist(y_train, num_bins, facecolor='green')
ax0_train.set_xlabel('Class ID')
ax0_train.set_ylabel('no. of images')
ax0_train.set_title('Distribution of train images')
n1_train, bins1_train, patches1_train = ax1_train.hist(y_train, num_bins, facecolor='green', normed=1)
ax1_train.set_xlabel('Class ID')
ax1_train.set_ylabel('Probability')
ax1_train.set_title('Normalized')
plt.show()

fig_valid, axes_valid = plt.subplots(nrows=1, ncols=2)
ax0_valid, ax1_valid = axes_valid.flatten()
n0_valid, bins0_valid, patches0_valid = ax0_valid.hist(y_valid, num_bins, facecolor='green')
ax0_valid.set_xlabel('Class ID')
ax0_valid.set_ylabel('no. of images')
ax0_valid.set_title('Distribution of validation images')
n1_valid, bins1_valid, patches1_valid = ax1_valid.hist(y_valid, num_bins, facecolor='green', normed=1)
ax1_valid.set_xlabel('Class ID')
ax1_valid.set_ylabel('Probability')
ax1_valid.set_title('Normalized')
plt.show()

fig_test, axes_test = plt.subplots(nrows=1, ncols=2)
ax0_test, ax1_test = axes_test.flatten()
n0_test, bins0_test, patches0_test = ax0_test.hist(y_test, num_bins, facecolor='green')
ax0_test.set_xlabel('Class ID')
ax0_test.set_ylabel('no. of images')
ax0_test.set_title('Distribution of test images')
n1_test, bins1_test, patches1_test = ax1_test.hist(y_test, num_bins, facecolor='green', normed=1)
ax1_test.set_xlabel('Class ID')
ax1_test.set_ylabel('Probability')
ax1_test.set_title('Normalized')
plt.show()

ratio_valid_train = np.mean(n1_valid/n1_train)
ratio_test_train = np.mean(n1_test/n1_train)

print("Ratio of normalized class distribution of valid to training set : ", ratio_valid_train)
print("Ratio of normalized class distribution of test to training set : ", ratio_test_train)

p_valid = (n_validation/n_train) * 100
p_test = (n_test/n_train) * 100

print("Size of validation set is {:.1f}% of training set".format(p_valid))
print("Size of test set is {:.1f}% of training set".format(p_test))

'''
1. Images with class id 20 seems to be more than other.
    These are commonly occuring traffic signs and it is necessary to train the model to these commonly occuring traffic
    signs.
2. All three sets seem to contain the same ratio of different class id. Hence the trained model's performance can be directly
    comparable to the validation and test sets. This has been verified with a normalized histogram.
'''