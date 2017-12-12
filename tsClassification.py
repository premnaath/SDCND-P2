# Load pickled data
import pickle
import numpy as np
import cv2

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

'''
image = plt.imread('./from_web/yield_crop.png')
X_image = np.pad(image, ((4, 0), (0, 0), (0, 0)), 'constant')
X_image = X_image[np.newaxis,:,:,:]

y_image = [13]
'''
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


# Pre-processing
from sklearn.utils import shuffle

# 1. Convert to grayscale
def convert_to_grayscale(x):
    grayscale_convert = []

    for i in range(len(x)):
        gray = cv2.cvtColor(x[i], cv2.COLOR_RGB2GRAY)
        grayscale_convert.append(gray)

    grayscale_convert = np.array(grayscale_convert)
    grayscale_convert = grayscale_convert[:,:,:,np.newaxis]

    return grayscale_convert


# 2. Convert to grayscale - method 2
def convert_to_grayscale_2(x):
    grayscale_convert = []

    for i in range(len(x)):
        gray = np.mean(x[i], axis=2)
        grayscale_convert.append(gray)

    grayscale_convert = np.array(grayscale_convert)
    grayscale_convert = grayscale_convert[:,:,:,np.newaxis]

    return grayscale_convert

# 3. Convert to grayscale with normalization - method 3
def convert_to_grayscale_3(x):
    grayscale_convert = []

    for i in range(len(x)):
        gray = np.mean(x[i], axis=2)
        mu = np.mean(gray)
        sigma = np.std(gray)
        gray_normalized = (gray - mu)/sigma
        grayscale_convert.append(gray_normalized)

    grayscale_convert = np.array(grayscale_convert)
    grayscale_convert = grayscale_convert[:,:,:,np.newaxis]

    return grayscale_convert

def normalize_single(img):
    img_norm = []

    for i in range(len(img.shape)):
        mu = np.mean(img[:,:,i])
        std = np.std(img[:,:,i])
        data = (img[:,:,i] - mu)/std
        img_norm.append(data)

    img_norm = np.array(img_norm)
    img_norm = img_norm.reshape(img.shape)

    return img_norm

def norm_each_channel(x):
    norm_img = []
    for i in range(len(x)):
        img = normalize_single(x[i])
        norm_img.append(img)

    norm_img = np.array(norm_img)
    return norm_img

# Model architecture
# Currently implementing the LeNet-6 architecture.

import tensorflow as tf
from tsLeNet import tfLeNet

EPOCHS = 10
BATCH_SIZE = 64
dropout = 0.5


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
keep_prob = tf.placeholder(tf.float32)

rate = 0.001

logits = tfLeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        X_train_pp = convert_to_grayscale_3(X_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_pp[offset:end], y_train[offset:end]
            #batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        X_train_pp = convert_to_grayscale_3(X_train)
        training_accuracy = evaluate(X_train_pp, y_train)
        print("EPOCH {} ...".format(i + 1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))

        X_valid_pp = convert_to_grayscale_3(X_valid)
        validation_accuracy = evaluate(X_valid_pp, y_valid)
        #validation_accuracy = evaluate(X_valid, y_valid)
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    X_test_pp = convert_to_grayscale_3(X_test)
    batch_x, batch_y = X_test_pp, y_test
    #batch_x, batch_y = X_test, y_test
    test_acc = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))

    X_image_pp = convert_to_grayscale_3(X_webtest)
    batch_x, batch_y = X_image_pp, y_webtest
    #batch_x, batch_y = X_image, y_image
    test_image_acc = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
    print('Testing Accuracy of image from web: {}'.format(test_image_acc))

    saver.save(sess, './P2.ckpt')
    print("Model saved")



