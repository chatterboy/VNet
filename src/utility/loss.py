import tensorflow as tf

def dice_coef(logits, labels):
    logits = tf.reshape(logits, [-1, 2])
    labels = tf.reshape(labels, [-1, 2])
    numerator = 2 * tf.reduce_sum(logits * labels)
    denominator = tf.reduce_sum(tf.square(logits) + tf.square(labels))
    epsilon = 0.00001
    dice_coef = numerator / (denominator + epsilon)
    return dice_coef

# https://github.com/Mazecreator/tensorflow-hints/tree/master/maximize
# https://stackoverflow.com/questions/38235648/is-there-an-easy-way-to-implement-a-optimizer-maximize-function-in-tensorflow
def dice_loss(logits, labels):
    with tf.variable_scope('dice_loss'):
        return 1 - dice_coef(logits, labels)