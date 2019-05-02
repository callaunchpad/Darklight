import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy

class Squeeze_UNet():
    def __init__(self, start_channel_depth=32, learning_rate=1e-3):
        """
        Builds the U-Net Computation graph
        :param start_channel_depth: the start channel depth that we change for benchmarking;
        default is the original architecture
        """
        print("Building model with starting channel depth {0}".format(start_channel_depth))
        self.build_model(start_channel_depth, learning_rate=learning_rate)

    def build_model(self, start_channel_depth, learning_rate=1e-3):

        def lrelu(x):
            return tf.maximum(x * 0.2, x)

        def fire_module(input, squeeze_output, expand_output, scopes):
            """channel axis is 3 i think for 4d"""
            #conv2d with 1x1
            #batch_normal
            input = slim.conv2d(input, squeeze_output, [1, 1], rate=1, activation_fn=lrelu, padding='same', scope=scopes[0])
            input = slim.batch_norm(input)

            #conv2d with 1x1
            #conv2d with 3x3
            left = slim.conv2d(input, expand_output // 2, [1, 1], rate=1, activation_fn=lrelu, padding='same', scope=scopes[1])
            right = slim.conv2d(input, expand_output - expand_output // 2, [3, 3], rate=1, activation_fn=lrelu, padding='same', scope=scopes[2])
            #concat
            result = tf.concat([left, right], axis=-1)
            return result
        # The tf session we're working in
        tf.reset_default_graph()

        # Input placeholder
        self.input = tf.placeholder(shape=[None, None, None, 4], dtype=tf.float32, name='Inputs')
        self.labels = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name='Labels')

        # The following is from https://bit.ly/2UAvptW
        def upsample_and_concat(x1, x2, output_channels, in_channels):
            pool_size = 2
            deconv_filter = tf.Variable(
                tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
            deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

            deconv_output = tf.concat([deconv, x2], 3)
            deconv_output.set_shape([None, None, None, output_channels * 2])

            return deconv_output

        conv1 = fire_module(self.input, 16, 64, scopes=['g_conv1_fm1_squeeze', 'g_conv1_fm1_left', 'g_conv1_fm1_right'])
        print(conv1.get_shape())
        conv1 = fire_module(conv1, 16, 64, scopes=['g_conv1_fm2_squeeze', 'g_conv1_fm2_left', 'g_conv1_fm2_right'])
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')
        print(conv1.get_shape())
        print(pool1.get_shape())
        print(tf.shape(pool1))

        conv2 = fire_module(pool1, 32, 128, scopes=['g_conv2_fm1_squeeze', 'g_conv2_fm1_left', 'g_conv2_fm1_right'])
        conv2 = fire_module(conv2, 32, 128, scopes=['g_conv2_fm2_squeeze', 'g_conv2_fm2_left', 'g_conv2_fm2_right'])
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')
        print(tf.shape(conv2))
        print(tf.shape(pool2))

        conv3 = fire_module(pool2, 48, 192, scopes=['g_conv3_fm1_squeeze', 'g_conv3_fm1_left', 'g_conv3_fm1_right'])
        conv3 = fire_module(conv3, 48, 192, scopes=['g_conv3_fm2_squeeze', 'g_conv3_fm2_left', 'g_conv3_fm2_right'])
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')
        print(tf.shape(conv3))
        print(tf.shape(pool3))

        conv4 = fire_module(pool3, 64, 256, scopes=['g_conv4_fm1_squeeze', 'g_conv4_fm1_left', 'g_conv4_fm1_right'])
        conv4 = fire_module(conv4, 64, 256, scopes=['g_conv4_fm2_squeeze', 'g_conv4_fm2_left', 'g_conv4_fm2_right'])
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')
        print(tf.shape(conv4))
        print(tf.shape(pool4))

        conv5 = fire_module(pool4, 80, 320, scopes=['g_conv5_fm1_squeeze', 'g_conv5_fm1_left', 'g_conv5_fm1_right'])
        conv5 = fire_module(conv5, 80, 320, scopes=['g_conv5_fm2_squeeze', 'g_conv5_fm2_left', 'g_conv5_fm2_right'])
        print(tf.shape(conv5))

        # out chan in chan
        up6 = upsample_and_concat(conv5, conv4, 256, 320)
        conv6 = fire_module(up6, 64, 256, scopes=['g_conv6_fm1_squeeze', 'g_conv6_fm1_left', 'g_conv6_fm1_right'])
        conv6 = fire_module(conv6, 64, 256, scopes=['g_conv6_fm2_squeeze', 'g_conv6_fm2_left', 'g_conv6_fm2_right'])

        up7 = upsample_and_concat(conv6, conv3, 192, 256)
        conv7 = fire_module(up7, 48, 192, scopes=['g_conv7_fm1_squeeze', 'g_conv7_fm1_left', 'g_conv7_fm1_right'])
        conv7 = fire_module(conv7, 48, 192, scopes=['g_conv7_fm2_squeeze', 'g_conv7_fm2_left', 'g_conv7_fm2_right'])

        up8 = upsample_and_concat(conv7, conv2, 128, 192)
        conv8 = fire_module(up8, 32, 128, scopes=['g_conv8_fm1_squeeze', 'g_conv8_fm1_left', 'g_conv8_fm1_right'])
        conv8 = fire_module(conv8, 32, 128, scopes=['g_conv8_fm2_squeeze', 'g_conv8_fm2_left', 'g_conv8_fm2_right'])

        up9 = upsample_and_concat(conv8, conv1, 64, 128)
        conv9 = fire_module(up9, 16, 64, scopes=['g_conv9_fm1_squeeze', 'g_conv9_fm1_left', 'g_conv9_fm1_right'])
        conv9 = fire_module(conv9, 16, 64, scopes=['g_conv9_fm2_squeeze', 'g_conv9_fm2_left', 'g_conv9_fm2_right'])

        conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
        out = tf.depth_to_space(conv10, 2, name="network_output")

        self.output = out

        # The loss, optimizer, and training op
        self.loss = tf.reduce_mean(tf.abs(self.output - self.labels ))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        # Create session and build parameters
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def train_step(self, x, y, sess):
        """
        Takes a training step on the batch fed in using the session given
        :param x: The input batch
        :param y: The output batch
        :param sess: The session to run this in
        :return: The value of the loss for this training step
        """
        feed_dict = {
            self.input: x,
            self.labels: y
        }

        loss_value, _ = sess.run((self.loss, self.train_op), feed_dict=feed_dict)

        return loss_value

    def evaluate(self, x, y, sess):
        """
        Computes the loss on the batch passed in
        :param x: The input batch for this evaluation
        :param y: The labels batch for this evaluation
        :param sess: The session in which to run this
        :return: The value of the loss on this batch
        """
        feed_dict = {
            self.input: x,
            self.labels: y
        }

        loss_value = sess.run(self.loss, feed_dict=feed_dict)

        return loss_value

    def predict(self, x, sess):
        """
        Predicts the output image (batch) on the given input image (batch)
        :param x: The input batch
        :param sess: The session to run this in
        :return: The result of the forward pass through the network (outputted images)
        """
        feed_dict = {
            self.input: x
        }

        return sess.run(self.output, feed_dict=feed_dict)

        def save_model(self):
        """
        Saves the model in the checkpoints folder
        :return: None
        """
        print("Saving model...")
        saver = tf.train.Saver()
        saver.save(self.sess, "./checkpoints/SQ_UNet" + str(self.start_channel_depth))

    def load_model(self, starting_depth):
        """
        Loads in the pre-trained weights from the specified model
        :param starting_depth: Specifies a model to load by the starting channel depth
        :return: None
        """
        # The saver to load the weights
        saver = tf.train.Saver()
        saver.restore(self.sess, "./checkpoints/SQ_UNet" + str(starting_depth))
