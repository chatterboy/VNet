import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.utility.dm import Batch, DataManager
from src.utility.da import DataAugmentation
from src.utility.loss import dice_loss
from src.utility.layer import conv3d, deconv3d, set_bsz

class VNet:
    def __init__(self, config):
        self.config = config
        self.depth = 64
        self.height = 128
        self.width = 128
        self.nclass = 2

        set_bsz(self.config['batch_size'])

    def to_one_hot(self, numpy_labels, dtype=np.float32):
        """
        :param numpy_labels: list of numpy_labels, a list of numpy_labels
        :param dtype:
        :return: new_numpy_labels: 5-D numpy, [batch_size, depth, height, width, 2]
        """
        batch_size = len(numpy_labels)
        depth, height, width = numpy_labels[0].shape
        new_numpy_labels = np.zeros((batch_size, depth, height, width, 2), dtype=dtype)
        for b in range(batch_size):
            for d in range(depth):
                for h in range(height):
                    for w in range(width):
                        if numpy_labels[b][d, h, w] < 0.5:
                            new_numpy_labels[b, d, h, w, 1] = 0
                            new_numpy_labels[b, d, h, w, 0] = 1
                        else:
                            new_numpy_labels[b, d, h, w, 1] = 1
                            new_numpy_labels[b, d, h, w, 0] = 0
        return new_numpy_labels

    def get_batch(self, numpy_images, numpy_labels, batch_size):
        return Batch(numpy_images, numpy_labels, batch_size)

    def preprocess(self):
        dm = DataManager(self.config)
        dm.load_train_data()
        # All images and labels need to have rescaling and resampling before doing next augmenting
        da = DataAugmentation()
        self.sitk_images = da.resample_sitk_images_to_desired_size(da.rescale_sitk_images(dm.get_sitk_images()), [128, 128, 64])
        self.sitk_labels = da.resample_sitk_images_to_desired_size(da.rescale_sitk_images(dm.get_sitk_labels()), [128, 128, 64])
        # original images
        self.numpy_images = dm.get_numpy_from_sitk_images(self.sitk_images)
        self.numpy_labels = dm.get_numpy_from_sitk_images(self.sitk_labels)
        print("Applying standard histogram equalization to numpy_images and numpy_labels")
        self.numpy_images += da.apply_hist_eqaulization_to_numpy_images(dm.get_numpy_from_sitk_images(self.sitk_images))
        self.numpy_labels += dm.get_numpy_from_sitk_images(self.sitk_labels)
        print("Total number of numpy_images and numpy_labels to give for the model to train: ({}, {})".format(len(self.numpy_images), len(self.numpy_labels)))
        self.numpy_images = np.asarray(self.numpy_images, dtype=np.float32).reshape(-1, self.depth, self.height, self.width, 1)
        self.numpy_labels = self.to_one_hot(self.numpy_labels, dtype=np.float32)
        temp_path = os.path.join(self.config['base_path'], self.config['temp_path'])
        self.save_numpy_images(self.numpy_images, 'images', temp_path)
        self.save_numpy_images(self.numpy_labels, 'labels', temp_path)
        print("Done with data augmentation")

    def _save_numpy_image_type_1(self, numpy_image, name, path):
        depth, _, _ = numpy_image.shape
        fig = plt.figure(figsize=(30, 30))
        cols = 8
        rows = depth / cols + 1
        for i in range(depth):
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(numpy_image[i], cmap='gray')
        plt.savefig(os.path.join(path, ''.join([name, '.png'])))
        plt.close()

    def _save_numpy_images_type_1(self, numpy_images, name, config):
        for index, numpy_image in enumerate(numpy_images):
            self._save_numpy_image_type_1(numpy_image, ''.join([name, '-', str(index)]), config)

    def _save_numpy_images_without_channels(self, numpy_images, name, path):
        new_numpy_images = []
        for b in range(numpy_images.shape[0]):
            new_numpy_images.append(numpy_images[b, :, :, :])
        self._save_numpy_images_type_1(new_numpy_images, name, path)

    def _save_numpy_images_type_2(self, numpy_images, name, path):
        self._save_numpy_images_without_channels(numpy_images[:, :, :, :, 0], name, path)

    def _save_numpy_images_type_3(self, numpy_images, name, path):
        self._save_numpy_images_without_channels(numpy_images[:, :, :, :, 0], ''.join([name, '-', 'ch0']), path)
        self._save_numpy_images_without_channels(numpy_images[:, :, :, :, 1], ''.join([name, '-', 'ch1']), path)

    def _save_numpy_images(self, numpy_images, name, path):
        """
            There are some types for numpy-type images in this model
                1. a list of 3-D numpy arrays
                2. 5-D numpy arrays, [batch_size, depth, height, width, 1]
                3. 5-D numpy arrays, [batch_size, depth, height, width, 2]
            1:  This is an essential type, a list consists of numpy arrays.
                A list is corresponding to a total size of dataset such as
                A list => [numpy array 1, numpy array 2, ...]
            2:  This type is reshaped from the type (1). Because this type
                of data is used to train this model. It can be train dataset
            3:  This type is transformed from the type (1) via to_one_hot().
                It can be ground-truth dataset.
        :param numpy_images: list of numpy, a list of numpy_images
        :param name:
        :param path:
        :return:
        """
        if type(numpy_images).__name__ == 'list':
            self._save_numpy_images_type_1(numpy_images, name, path)
        else:
            if numpy_images.shape[-1] == 1:
                self._save_numpy_images_type_2(numpy_images, name, path)
            else:
                self._save_numpy_images_type_3(numpy_images, name, path)

    def save_numpy_images(self, numpy_images, name, path):
        self._save_numpy_images(numpy_images, name, path)

    def train(self):
        self.preprocess()
        batch = self.get_batch(self.numpy_images, self.numpy_labels, self.config['batch_size'])
        x = tf.placeholder(tf.float32, shape=[
            self.config['batch_size'],
            self.depth, self.height, self.width,
            1], name='x')
        y = tf.placeholder(tf.float32, shape=[
            self.config['batch_size'],
            self.depth, self.height, self.width,
            self.nclass], name='y')
        logits_op = self.vnet(x)
        loss_op = dice_loss(logits_op, y)
        #trainer_op = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate']).minimize(loss_op)
        trainer_op = tf.train.MomentumOptimizer(learning_rate=self.config['learning_rate'], momentum=self.config['momentum']).minimize(loss_op)
        tf.add_to_collection("trainer", trainer_op)
        saver = tf.train.Saver() # logit 위에 두면 variables 없다고 에러 뜸
        bestLoss = None
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for epoch in range(1, self.config['epochs'] + 1):
                batch_x, batch_y = batch.next_to()
                loss, _ = sess.run([loss_op, trainer_op],
                                   feed_dict={x: batch_x, y: batch_y})
                print('epoch: {}, loss: {}'.format(epoch, loss))
                # Save the best for current model info until now
                if bestLoss is None or bestLoss > loss:
                    saver.save(sess, os.path.join(self.config['base_path'],self.config['chks_path'], 'vnet'))
                    bestLoss = loss
                # NOTE: Now, this functionality is implemented using train data
                #       after that we will modify this using validation data
                # Check current model in validation
                # test with a validation data and need to look at something:
                #   1. the images that created by the model from the validation data
                if epoch % self.config['valid_step'] == 0:
                    logits, loss = sess.run([logits_op, loss_op],
                                            feed_dict={x: batch_x, y: batch_y})
                    print('validation: {}'.format(loss))
                    result_path = os.path.join(self.config['base_path'], self.config['result_path'])
                    self.save_numpy_images(logits, ''.join([str(epoch), '-', 'logits']), result_path)
                    self.save_numpy_images(batch_x, ''.join([str(epoch), '-', 'batch_x']), result_path)
                    self.save_numpy_images(batch_y, ''.join([str(epoch), '-', 'batch_y']), result_path)

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
            loss = dice_loss(logit, y)
            trainer = tf.train.MomentumOptimizer(learning_rate=self.config['learning_rate'],
                                                 momentum=self.config['momentum']).minimize(loss)

            for epoch in range(1, self.config['epochs'] + 1):
                batch_x, batch_y = batch.nextTo()

                cost, _ = sess.run([loss, trainer], feed_dict={x: batch_x,
                                                               y: batch_y})


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
            # logits
            #       return : 5-D Tensor, [batch size, 64, 128, 128, 2]
            logits = conv3d('logits', r9, 2, 1)
            logits = tf.nn.softmax(logits, name='softmax')
            print('logit: {}'.format(logits.get_shape()))

            return logits