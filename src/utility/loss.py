import tensorflow as tf

def diceCoef(logit, label):
    logit = tf.reshape(logit, [-1, 2])
    label = tf.reshape(label, [-1, 2])
    a = 2 * tf.reduce_sum(tf.multiply(logit, label))
    b = tf.reduce_sum(tf.square(logit) + tf.square(label))
    return tf.reduce_mean(tf.div(a, b))

def diceLoss(logit, label):
    with tf.variable_scope('dice_loss'):
        return 1 - diceCoef(logit, label)