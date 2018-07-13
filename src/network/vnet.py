import os
import numpy as np
import tensorflow as tf

from src.utility.dm import Batch, DataManager
from src.utility.loss import diceLoss
from src.utility.layer import conv3d, deconv3d, set_bsz

class VNet:
    def __init__(self, config):
        self.config = config
        self.depth = 64
        self.height = 128
        self.width = 128
        self.nclass = 2

        set_bsz(self.config['batch_size'])

    def toOneHot(self, labels):
        # TODO: Need to extend generally
        shape = labels.shape
        ret = np.zeros((shape[0], shape[1], shape[2], shape[3], 2), dtype=np.int32)
        print('start to transform labels in one-hot type...')
        for b in range(shape[0]):
            for i in range(shape[1]):
                for j in range(shape[2]):
                    for k in range(shape[3]):
                        if labels[b, i, j, k] > 0.5:
                            ret[b, i, j, k, 1] = 1
                            ret[b, i, j, k, 0] = 0
                        else:
                            ret[b, i, j, k, 1] = 0
                            ret[b, i, j, k, 0] = 1
            print('label {} is done...'.format(b))
        return ret

    def loadTrain(self):
        DM = DataManager(self.config)

        DM.load_train()
        self.images = np.asarray(DM.get_numpy_images(),
                                 dtype=np.float32).reshape((-1, self.depth,
                                                            self.height, self.width, 1))
        self.labels = self.toOneHot(np.asarray(DM.get_numpy_labels(), dtype=np.float32))

    def train(self):
        print('lodaing train dataset...')
        self.loadTrain()

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

        saver = tf.train.Saver()

        logit = self.vnet(x)

        loss = diceLoss(logit, y)

        trainer = tf.train.MomentumOptimizer(learning_rate=self.config['learning_rate'],
                                             momentum=self.config['momentum']).minimize(loss)
        tf.add_to_collection("trainer", trainer)

        bestCost = None

        print('start to train VNet...')
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            for epoch in range(1, self.config['epochs'] + 1):
                batch_x, batch_y = batch.nextTo()

                # cost: numpy.float32
                cost, _ = sess.run([loss, trainer], feed_dict={x: batch_x,
                                                               y: batch_y})

                print('epoch: {}, cost: {}'.format(epoch, cost))

                if bestCost is None or bestCost > cost:
                    saver.save(sess, os.path.join(self.config['base_path'],
                                                  self.config['chks_path'], 'vnet'))
                    bestCost = cost

    def retrain(self):
        # checkpoint 경로 생성
        chksPath = os.path.join(self.config['base_path'], self.config['chks_path'])
        # chekcpoint 디렉토리 확인
        assert os.path.exists(chksPath), 'There is no such a checkpoint directory.'

        print('lodaing train dataset...')
        self.loadTrain()

        batch = Batch(self.images, self.labels, batch_size=2)

        with tf.Session() as sess:
            # 모델 및 학습된 가중치 로드
            saver = tf.train.import_meta_graph(os.path.join(chksPath, 'vnet.meta'))
            saver.restore(sess, tf.train.latest_checkpoint(chksPath))
            graph = tf.get_default_graph()

            for op in tf.get_default_graph().get_operations():
                print(op.name)

            # 학습
            x = graph.get_tensor_by_name('x:0')
            y = graph.get_tensor_by_name('y:0')

            logit = graph.get_tensor_by_name('VNet/logits/add:0')
            loss = diceLoss(logit, y)
            trainer = tf.train.MomentumOptimizer(learning_rate=self.config['learning_rate'],
                                                 momentum=self.config['momentum']).minimize(loss)

            for epoch in range(1, self.config['epochs'] + 1):
                batch_x, batch_y = batch.nextTo()

                cost, _ = sess.run([loss, trainer], feed_dict={x: batch_x,
                                                               y: batch_y})

                if epoch % self.config['epoch_step'] == 0:
                    print('epoch: {}, cost: {}'.format(epoch, cost))

    def test(self):
        return

    def vnet(self, x):
        with tf.variable_scope('VNet'):
            # Stage 1
            # x : 5-D Tensor, [batch size, depth, height, width, channels]
            #       return : 5-D Tensor, [batch size, 64, 128, 128, 1]
            # conv1_1 : 3-D convolution layer, 3x3x3@16
            #       return : 5-D Tensor, [batch size, 64, 128, 128, 16]
            conv1_1 = conv3d('conv1_1', x, 16, 3, relu=tf.nn.relu)
            print('conv1_1: {}'.format(conv1_1.get_shape()))
            # r1 : residual layer, [batch size, 64, 128, 128, 16] + [batch size, 64, 128, 128, 1]
            #       return : 5-D Tensor, [batch size, 64, 128, 128, 16]
            r1 = tf.add(conv1_1, x, 'r1')
            print('r1: {}'.format(r1.get_shape()))
            # down_conv1 : 3-D convolution layer, 2x2x2@32,2
            #       return : 5-D Tensor, [batch size, 32, 64, 64, 32]
            down_conv1 = conv3d('down_conv1', r1, 32, 2, 2, relu=tf.nn.relu)
            print('down_conv1: {}'.format(down_conv1.get_shape()))

            # Stage 2
            # conv2_1 : 3-D convolution layer, 3x3x3@32
            #       return : 5-D Tensor, [batch size, 32, 64, 64, 32]
            conv2_1 = conv3d('conv2_1', down_conv1, 32, 3, relu=tf.nn.relu)
            print('conv2_1: {}'.format(conv2_1.get_shape()))
            # conv2_2 : 3-D convolution layer, 3x3x3@32
            #       return : 5-D Tensor, [batch size, 32, 64, 64, 32]
            conv2_2 = conv3d('conv2_2', conv2_1, 32, 3, relu=tf.nn.relu)
            print('conv2_2: {}'.format(conv2_2.get_shape()))
            # r2 : down_conv1 + conv2_2
            #       down_conv1 : [batch size, 32, 64, 64, 32]
            #       conv2_2 : [batch size, 32, 64, 64, 32]
            #       return : 5-D Tensor, [batch size, 32, 64, 64, 32]
            r2 = tf.add(conv2_2, down_conv1, 'r2')
            print('r2: {}'.format(r2.get_shape()))
            # down_conv2 : 3-D convolution layer, 2x2x2@64
            #       return : 5-D Tensor, [batch size, 16, 32, 32, 64]
            down_conv2 = conv3d('down_conv2', r2, 64, 2, 2, relu=tf.nn.relu)
            print('down_conv2: {}'.format(down_conv2.get_shape()))

            # Stage 3
            # conv3_1 : 3-D convolution layer, 3x3x3@64
            #       return : 5-D Tensor, [batch size, 16, 32, 32, 64]
            conv3_1 = conv3d('conv3_1', down_conv2, 64, 3, relu=tf.nn.relu)
            print('conv3_1: {}'.format(conv3_1.get_shape()))
            # conv3_2 : 3-D convolution layer, 3x3x3@64
            #       return : 5-D Tensor, [batch size, 16, 32, 32, 64]
            conv3_2 = conv3d('conv3_2', conv3_1, 64, 3, relu=tf.nn.relu)
            print('conv3_2: {}'.format(conv3_2.get_shape()))
            # conv3_3 : 3-D convolution layer, 3x3x3@
            #       return : 5-D Tensor, [batch size, 16, 32, 32, 64]
            conv3_3 = conv3d('conv3_3', conv3_2, 64, 3, relu=tf.nn.relu)
            print('conv3_3: {}'.format(conv3_3.get_shape()))
            # r3 : down_conv2 + conv3_3
            #       return : 5-D Tensor, [batch size, 16, 32, 32, 64]
            r3 = tf.add(down_conv2, conv3_3, 'r3')
            print('r3: {}'.format(r3.get_shape()))
            # down_conv3 : 3-D convolution layer, 2x2x2@128
            #       return : 5-D, [batch size, 8, 16, 16, 128]
            down_conv3 = conv3d('down_conv3', r3, 128, 2, 2, relu=tf.nn.relu)
            print('down_conv3: {}'.format(down_conv3.get_shape()))

            # Stage 4
            # conv4_1 : 3-D convolution layer, 3x3x3@128
            #       return : 5-D Tensor, [batch size, 8, 16, 16, 128]
            conv4_1 = conv3d('conv4_1', down_conv3, 128, 3, relu=tf.nn.relu)
            print('conv4_1: {}'.format(conv4_1.get_shape()))
            # conv4_2 : 3-D convolution layer, 3x3x3@128
            #       return : 5-D Tensor, [batch size, 8, 16, 16, 128]
            conv4_2 = conv3d('conv4_2', conv4_1, 128, 3, relu=tf.nn.relu)
            print('conv4_2: {}'.format(conv4_2.get_shape()))
            # conv4_3 : 3-D convolution layer, 3x3x3@128
            #       return : 5-D Tensor, [batch size, 8, 16, 16, 128]
            conv4_3 = conv3d('conv4_3', conv4_2, 128, 3, relu=tf.nn.relu)
            print('conv4_3: {}'.format(conv4_3.get_shape()))
            # r4 : down_conv3 + conv4_3
            #       return : 5-D Tensor, [batch size, 8, 16, 16, 128]
            r4 = tf.add(down_conv3, conv4_3, 'r4')
            print('r4: {}'.format(r4.get_shape()))
            # down_conv4 : 3-D convolution layer, 2x2x2@256
            #       return : 5-D Tensor, [batch size, 4, 8, 8, 256]
            down_conv4 = conv3d('down_conv4', r4, 256, 2, 2, relu=tf.nn.relu)
            print('down_conv4: {}'.format(down_conv4.get_shape()))

            # Stage 5
            # conv5_1 : 3-D convolution layer, 3x3x3@256
            #       return : 5-D Tensor, [batch size, 4, 8, 8, 256]
            conv5_1 = conv3d('conv5_1', down_conv4, 256, 3, relu=tf.nn.relu)
            print('conv5_1: {}'.format(conv5_1.get_shape()))
            # conv5_2 : 3-D convolution layer, 3x3x3@256
            #       return : 5-D Tensor, [batch size, 4, 8, 8, 256]
            conv5_2 = conv3d('conv5_2', conv5_1, 256, 3, relu=tf.nn.relu)
            print('conv5_2: {}'.format(conv5_2.get_shape()))
            # conv5_3 : 3-D convolution layer, 3x3x3@256
            #       return : 5-D Tensor, [batch size, 4, 8, 8, 256]
            conv5_3 = conv3d('conv5_3', conv5_2, 256, 3, relu=tf.nn.relu)
            print('conv5_3: {}'.format(conv5_3.get_shape()))
            # r5 : down_conv4 + conv5_3
            #       return : 5-D Tensor, [batch size, 4, 8, 8, 256]
            r5 = tf.add(down_conv4, conv5_3, 'r5')
            print('r5: {}'.format(r5.get_shape()))
            # up_conv5
            #       return : 5-D Tensor, [batch size, 8, 16, 16, 256]
            up_conv5 = deconv3d('up_conv5', r5, 256, 2, relu=tf.nn.relu)
            print('up_conv5: {}'.format(up_conv5.get_shape()))

            # Stage 6
            # concat6 : r4 and up_conv5
            #       r4 : [batch size, 8, 16, 16, 128]
            #       up_conv5 : [batch size, 8, 16, 16, 256]
            #       return : 5-D Tensor, [batch size, 8, 16, 16, 384]
            concat6 = tf.concat([r4, up_conv5], axis=-1, name='concat6')
            print('concat6: {}'.format(concat6.get_shape()))
            # conv6_1 : 3-D convolution layer, 3x3x3@256
            #       return : 5-D Tensor, [batch size, 8, 16, 16, 256]
            conv6_1 = conv3d('conv6_1', concat6, 256, 3, relu=tf.nn.relu)
            print('conv6_1: {}'.format(conv6_1.get_shape()))
            # conv6_2 : 3-D convolution layer, 3x3x3@256
            #       return : 5-D Tensor, [batch size, 8, 16, 16, 256]
            conv6_2 = conv3d('conv6_2', conv6_1, 256, 3, relu=tf.nn.relu)
            print('conv6_2: {}'.format(conv6_2.get_shape()))
            # conv6_3 : 3-D convolution layer, 3x3x3@256
            #       return : 5-D Tensor, [batch size, 8, 16, 16, 256]
            conv6_3 = conv3d('conv6_3', conv6_2, 256, 3, relu=tf.nn.relu)
            print('conv6_3: {}'.format(conv6_3.get_shape()))
            # r6 : up_conv5 + conv6_3
            #       return : 5-D Tensor, [batch size, 8, 16, 16, 256]
            r6 = tf.add(up_conv5, conv6_3, 'r6')
            print('r6: {}'.format(r6.get_shape()))
            # up_conv6
            #       return : 5-D Tensor, [batch size, 16, 32, 32, 128]
            up_conv6 = deconv3d('up_conv6', r6, 128, 2, relu=tf.nn.relu)
            print('up_conv6: {}'.format(up_conv6.get_shape()))

            # Stage 7
            # concat7 : r3 and up_conv6
            #       r3 : [batch size, 16, 32, 32, 64]
            #       up_conv6 : [batch size, 16, 32, 32, 128]
            #       return : 5-D Tensor, [batch size, 16, 32, 32, 192]
            concat7 = tf.concat([r3, up_conv6], axis=-1, name='concat7')
            print('concat7: {}'.format(concat7.get_shape()))
            # conv7_1 : 3-D convolution layer, 3x3x3@128
            #       return : 5-D Tensor, [batch size, 16, 32, 32, 128]
            conv7_1 = conv3d('conv7_1', concat7, 128, 3, relu=tf.nn.relu)
            print('conv7_1: {}'.format(conv7_1.get_shape()))
            # conv7_2 : 3-D convolution layer, 3x3x3@128
            #       return : 5-D Tensor, [batch size, 16, 32, 32, 128]
            conv7_2 = conv3d('conv7_2', conv7_1, 128, 3, relu=tf.nn.relu)
            print('conv7_2: {}'.format(conv7_2.get_shape()))
            # conv7_3 : 3-D convolution layer, 3x3x3@128
            #       return : 5-D Tensor, [batch size, 16, 32, 32, 128]
            conv7_3 = conv3d('conv7_3', conv7_2, 128, 3, relu=tf.nn.relu)
            print('conv7_3: {}'.format(conv7_3.get_shape()))
            # r7 : up_conv6 + conv7_3
            #       return : 5-D Tensor, [batch size, 16, 32, 32, 128]
            r7 = tf.add(up_conv6, conv7_3, 'r7')
            print('r7: {}'.format(r7.get_shape()))
            # up_conv7
            #       return : 5-D Tensor, [batch size, 32, 64, 64, 64]
            up_conv7 = deconv3d('up_conv7', r7, 64, 2, relu=tf.nn.relu)
            print('up_conv7: {}'.format(up_conv7.get_shape()))

            # Stage 8
            # concat8 : r2 and up_conv7
            #       r2 : [batch size, 32, 64, 64, 32]
            #       up_conv7 : [batch size, 32, 64, 64, 64]
            #       return : 5-D Tensor, [batch size, 32, 64, 64, 96]
            concat8 = tf.concat([r2, up_conv7], axis=-1, name='concat8')
            print('concat8: {}'.format(concat8.get_shape()))
            # conv8_1
            #       return : 5-D Tensor, [batch size, 32, 64, 64, 64]
            conv8_1 = conv3d('conv8_1', concat8, 64, 3, relu=tf.nn.relu)
            print('conv8_1: {}'.format(conv8_1.get_shape()))
            # conv8_2
            #       return : 5-D Tensor, [batch size, 32, 64, 64, 64]
            conv8_2 = conv3d('conv8_2', conv8_1, 64, 3, relu=tf.nn.relu)
            print('conv8_2: {}'.format(conv8_2.get_shape()))
            # r8 : up_conv7 + conv8_2
            #       return : 5-D Tensor, [batch size, 32, 64, 64, 64]
            r8 = tf.add(up_conv7, conv8_2, 'r8')
            print('r8: {}'.format(r8.get_shape()))
            # up_conv8
            #       return : 5-D Tensor, [batch size, 64, 128, 128, 32]
            up_conv8 = deconv3d('up_conv8', r8, 32, 2, relu=tf.nn.relu)
            print('up_conv8: {}'.format(up_conv8.get_shape()))

            # Stage 9
            # concat9 : r1 and up_conv8
            #       r1 : [batch size, 64, 128, 128, 16]
            #       up_conv8 : [batch size, 64, 128, 128, 32]
            #       return : 5-D Tensor, [batch size, 64, 128, 128, 48]
            concat9 = tf.concat([r1, up_conv8], axis=-1, name='concat9')
            print('concat9: {}'.format(concat9.get_shape()))
            # conv9_1 : 3-D convolution layer, 3x3x3@32
            #       return : 5-D Tensor, [batch size, 64, 128, 128, 32]
            conv9_1 = conv3d('conv9_1', concat9, 32, 3, relu=tf.nn.relu)
            print('conv9_1: {}'.format(conv9_1.get_shape()))
            # r9 : up_conv8 + conv9_1
            #       up_conv8 : [batch size, 64, 128, 128, 32]
            #       conv9_1 : [batch size, 64, 128, 128, 32]
            #       return : 5-D Tensor, [batch size, 64, 128, 128, 32]
            r9 = tf.add(up_conv8, conv9_1, 'r9')
            print('r9: {}'.format(r9.get_shape()))
            # TODO: score 레이어에서 마지막에 activation을 적용하는가? 우선은 적용함 (논문에서도 그래서)
            # logits
            #       return : 5-D Tensor, [batch size, 64, 128, 128, 2]
            logit = conv3d('logits', r9, 2, 1)
            print('logit: {}'.format(logit.get_shape()))

            return logit