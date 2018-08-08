import tensorflow as tf

BATCH_SIZE = 2

def set_bsz(bsz):
    global BATCH_SIZE
    BATCH_SIZE = bsz

def _prelu(name, features, alpha=0.2):
    with tf.variable_scope(name):
        a = tf.get_variable('a', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(alpha))
        return max(0, features) + a * min(0, features)

def prelu(name, features, alpha=0.2):
    with tf.variable_scope(name):
        a = tf.get_variable('a', shape=[1], dtype=tf.float32, initializer=tf.constant_initializer(alpha))
        return tf.nn.leaky_relu(features, a, 'prelu')

def conv3d(name, input, filter, ksz, stride=1, padding='SAME', relu=None):
    with tf.variable_scope(name):
        iShape = input.get_shape().as_list()
        kShape = [1] * 5
        kShape[1:4] = [ksz] * 3 if type(ksz).__name__ == 'int' else ksz
        sShape = [1] * 5
        sShape[1:4] = [stride] * 3 if type(stride).__name__ == 'int' else stride
        W = tf.get_variable('W', shape=[kShape[1], kShape[2], kShape[3], iShape[-1], filter],
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[filter], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        out = tf.nn.conv3d(input, filter=W, strides=sShape, padding=padding) + b
        if relu:
            return relu(out)
        return out

def deconv3d(name, input, filter, ksz, stride=2, padding='SAME', relu=None):
    with tf.variable_scope(name):
        iShape = input.get_shape().as_list()
        kShape = [1] * 5
        kShape[1:4] = [ksz] * 3 if type(ksz).__name__ == 'int' else ksz
        sShape = [1] * 5
        sShape[1:4] = [stride] * 3 if type(stride).__name__ == 'int' else stride
        W = tf.get_variable('W', shape=[kShape[1], kShape[2], kShape[3], filter, iShape[-1]],
                            dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[filter], dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())
        out = tf.nn.conv3d_transpose(input, filter=W,
                                     output_shape=tf.stack([BATCH_SIZE, sShape[1] * iShape[1],
                                                             sShape[2] * iShape[2],
                                                             sShape[3] * iShape[3], filter]),
                                     strides=sShape, padding=padding) + b
        if relu:
            return relu(out)
        return out