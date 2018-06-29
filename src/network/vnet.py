import os
import tensorflow as tf

from src.utility.dm import Batch
from src.utility.loss import dice_loss


class VNet:
    def __init__(self, config):
        self.config = config
        self.depth = 64
        self.height = 128
        self.width = 128
        self.nclass = 2

    def train(self):
        batch = Batch(self.images, self.labels, batch_size=2)

        x = tf.placeholder(tf.float32, shape=[self.config['batch_size'],
                                              self.depth,
                                              self.height,
                                              self.width,
                                              1], name='x')
        y = tf.placeholder(tf.float32, shape=[self.config['batch_size'],
                                              self.depth,
                                              self.height,
                                              self.width,
                                              self.nclass], name='y')

        logits = self.vnet()
        loss = dice_loss(logits, y)
        trainer = tf.train.MomentumOptimizer(learning_rate=self.config['learning_rate'],
                                             momentum=self.config['momentum']).minimize(loss)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            saver.save(sess, os.path.join(self.config['base_path'],
                                          self.config['chks_path'],
                                          'vnet'))

            for epoch in range(self.config['epochs']):
                batch_x, batch_y = batch.nextTo()

                cost, _ = sess.run([loss, trainer], feed_dict={x: batch_x,
                                                               y: batch_y})




        return

    def test(self):
        return