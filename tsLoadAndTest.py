import pickle
import numpy as np
import tensorflow as tf
from tsLeNet import tfLeNet

# TODO: Fill this in based on where you saved the training and testing data

testing_file = './traffic-signs-data/test.p'
web_testing_file = 'web_img.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
with open(web_testing_file, mode='rb') as f:
    webtest = pickle.load(f)

X_test, y_test = test['features'], test['labels']
X_webtest, y_webtest = webtest['features'], webtest['labels']


# Convert to grayscale with normalization - Pre-processing
def preProcessData(x):
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

EPOCHS = 10
BATCH_SIZE = 64
dropout = 0.75

tf.reset_default_graph()

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

logits = tfLeNet(x, keep_prob)
softmax = tf.nn.softmax(logits)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'P2.ckpt')

    X_test_pp = preProcessData(X_test)
    batch_x, batch_y = X_test_pp, y_test
    test_acc = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))

    X_image_pp = preProcessData(X_webtest)
    batch_x, batch_y = X_image_pp, y_webtest
    test_prediction = sess.run(correct_prediction, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
    print(test_prediction)
    test_image_acc = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
    print('Testing Accuracy of image from web: {}'.format(test_image_acc))
    softmax_prob = sess.run(softmax, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
    top_k = sess.run(tf.nn.top_k(softmax_prob, k=5))
    print(top_k)