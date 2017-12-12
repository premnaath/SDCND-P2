import tensorflow as tf

def tfLeNet(x, dropout):
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x8.
    outdepth1 = 8
    F_w1 = tf.Variable(tf.truncated_normal([5,5,1,outdepth1], mean=mu, stddev=sigma), name='w_convL1')
    F_b1 = tf.Variable(tf.zeros([outdepth1]), name='b_convL1')
    conv1 = tf.nn.conv2d(x, F_w1, strides=[1,1,1,1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, F_b1)

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x8. Output = 14x14x8.
    maxpool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x32.
    outdepth2 = 32
    F_w2 = tf.Variable(tf.truncated_normal([5,5,outdepth1,outdepth2], mean=mu, stddev=sigma), name='w_convL2')
    F_b2 = tf.Variable(tf.zeros([outdepth2]), name='b_convL2')
    conv2 = tf.nn.conv2d(maxpool1, F_w2, strides=[1,1,1,1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, F_b2)

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x32. Output = 5x5x32.
    maxpool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    # Flatten. Input = 5x5x32. Output = 800.
    flat = tf.contrib.layers.flatten(maxpool2)

    # Layer 3: Fully Connected. Input = 800. Output = 400.
    outdepth_fc1 = 400
    Fc_w1 = tf.Variable(tf.truncated_normal([800,outdepth_fc1], mean=mu, stddev=sigma), name='w_fcL1')
    Fc_b1 = tf.Variable(tf.zeros([outdepth_fc1]), name='b_fcL1')

    fc1 = tf.add(tf.matmul(flat, Fc_w1), Fc_b1)

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Layer 4: Fully Connected. Input = 400. Output = 200.
    outdepth_fc2 = 200
    Fc_w2 = tf.Variable(tf.truncated_normal([outdepth_fc1,outdepth_fc2], mean=mu, stddev=sigma), name='w_fcL2')
    Fc_b2 = tf.Variable(tf.zeros([outdepth_fc2]), name='b_fcL2')

    fc2 = tf.add(tf.matmul(fc1, Fc_w2), Fc_b2)

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # Layer 5: Fully Connected. Input = 200. Output = 84.
    outdepth_fc3 = 84
    Fc_w3 = tf.Variable(tf.truncated_normal([outdepth_fc2,outdepth_fc3], mean=mu, stddev=sigma), name='w_fcL3')
    Fc_b3 = tf.Variable(tf.zeros([outdepth_fc3]), name='b_fcL3')

    fc3 = tf.add(tf.matmul(fc2, Fc_w3), Fc_b3)

    # Activation
    fc3 = tf.nn.relu(fc3)

    # Dropout
    fc3 = tf.nn.dropout(fc3, dropout)

    # Layer 6: Fully Connected. Input = 84. Output = 43.
    outdepth_fc4 = 43
    Fc_w4 = tf.Variable(tf.truncated_normal([outdepth_fc3,outdepth_fc4], mean=mu, stddev=sigma), name='w_fcL4')
    Fc_b4 = tf.Variable(tf.zeros([outdepth_fc4]), name='b_fcL4')

    fc4 = tf.add(tf.matmul(fc3, Fc_w4), Fc_b4)

    return fc4