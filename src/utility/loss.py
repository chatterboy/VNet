import tensorflow as tf

def dice_coef(logits, labels):
    """
    :param logits: a 5-D tensor, [batch_size, depth, height, width, 2]
    :param labels: a 5-D tensor, [batch_size, depth, height, width, 2]
    :return:
    """
    logits = tf.reshape(logits, [2, -1])
    labels = tf.reshape(labels, [2, -1])
    numerator = tf.multiply(tf.constant(2, dtype=tf.float32), tf.reduce_sum(tf.multiply(logits, labels), axis=-1))
    denominator = tf.reduce_sum(tf.add(tf.square(logits), tf.square(labels)), axis=-1)
    dice_coef = tf.reduce_mean(tf.div(numerator, tf.add(denominator, tf.constant(0.00001, dtype=tf.float32))))
    return dice_coef

# https://github.com/Mazecreator/tensorflow-hints/tree/master/maximize
# https://stackoverflow.com/questions/38235648/is-there-an-easy-way-to-implement-a-optimizer-maximize-function-in-tensorflow
def dice_loss(logits, labels):
    with tf.variable_scope('dice_loss'):
        return 1 - dice_coef(logits, labels)