import tensorflow as tf

layers = tf.contrib.layers

# x is [None, ncontext, nbands, nch]
# c is [None, nfeats]
# y is [None]
def placement_cnn(x, c, y, dtype=dtype):
    # Convolutional layers
    conv_shapes = [(7, 3, 10), (3, 3, 20)]
    pool_shapes = [(1, 3), (1, 3)]
    for i, (conv_shape, pool_shape) in enumerate(zip(conv_shapes, pool_shapes)):
        with tf.variable_scope('conv_{}'.format(i)):
            x = layers.conv2d(x, conv_shape[2], conv_shape[:2], padding='VALID', activation_fn=tf.nn.relu)
            x = layers.max_pool2d(x, pool_shape, pool_shape)

    # Flatten
    x = tf.reshape(x, [None, -1])

    # Append feats
    x = tf.concat([x, c], axis=1)

    # Dense layers
    dense_sizes = [256, 128, 1]
    for i, size in enumerate(dense_sizes):
        if size == 1:
            activation_fn = None
        else:
            activation_fn = tf.nn.relu

        with tf.variable_scope('dense_{}'.format(i)):
            x = layers.fully_connected(x, size, activation_fn)

    yhat = tf.squeeze(x, axis=1)

    return yhat
