import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.tools import inspect_checkpoint as chkp
import numpy as np
from tensorflow.python import pywrap_tensorflow

class UNet():
    def __init__(self, start_channel_depth=32, learning_rate=1e-4, student=False, teacher_size=32):
        """
        Builds the U-Net Computation graph
        :param start_channel_depth: the start channel depth that we change for benchmarking;
        :param learning_rate: The learning rate to use for training
        :param student: Whether or not this is a student model
        :param teacher_size: The starting channel depth of the teacher UNet so we can match
        dimensions in the pieces of the network
        default is the original architecture
        """
        self.start_learning_rate = learning_rate
        self.student = student
        self.teacher_size = teacher_size
        self.start_channel_depth = start_channel_depth

        print(f"Building model with starting channel depth {start_channel_depth}")
        self.build_model(start_channel_depth, learning_rate=learning_rate)

    def build_model(self, start_channel_depth, learning_rate=1e-3):
        # The tf session we're working in
        with tf.device("/device:GPU:2"):
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

            # The first piece of the piecewise distillation
            with tf.variable_scope("piece0"):
                conv1 = slim.conv2d(self.input, start_channel_depth, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv1_1')
                conv1 = slim.conv2d(conv1, start_channel_depth, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv1_2')
                pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

                conv2 = slim.conv2d(pool1, start_channel_depth * 2, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv2_1')
                conv2 = slim.conv2d(conv2, start_channel_depth * 2, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv2_2')
                pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')
                # Add 1x1 convolution to match size of teacher model
                if self.student:
                    self.piece_1_out = slim.conv2d(pool2, self.teacher_size * 2, [1,1], scope='piece0_1x1')
                    self.piece_1_target = tf.placeholder(dtype=tf.float32, shape=self.piece_1_out.shape)
                    self.piece_1_loss = tf.losses.mean_squared_error(self.piece_1_target, self.piece_1_out)
                else:
                    self.piece_1_out = pool2

            # The second piece of the piecewise distillation
            with tf.variable_scope("piece1"):
                conv3 = slim.conv2d(self.piece_1_out, start_channel_depth * 4, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv3_1')
                conv3 = slim.conv2d(conv3, start_channel_depth * 4, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv3_2')
                pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

                conv4 = slim.conv2d(pool3, start_channel_depth * 8, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv4_1')
                conv4 = slim.conv2d(conv4, start_channel_depth * 8, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv4_2')
                pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

                conv5 = slim.conv2d(pool4, start_channel_depth * 16, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv5_1')
                conv5 = slim.conv2d(conv5, start_channel_depth * 16, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv5_2')

                up6 = upsample_and_concat(conv5, conv4, start_channel_depth * 8, start_channel_depth * 16)
                conv6 = slim.conv2d(up6, start_channel_depth * 8, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv6_1')
                conv6 = slim.conv2d(conv6, start_channel_depth * 8, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv6_2')

                up7 = upsample_and_concat(conv6, conv3, start_channel_depth * 4, start_channel_depth * 8)
                conv7 = slim.conv2d(up7, start_channel_depth * 4, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv7_1')
                conv7 = slim.conv2d(conv7, start_channel_depth * 4, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv7_2')

                up8 = upsample_and_concat(conv7, conv2, start_channel_depth * 2, start_channel_depth * 4)

                if self.student:
                    self.piece_2_out = slim.conv2d(up8, self.teacher_size * 2, [1,1], scope='piece1_1x1')
                    self.piece_2_target = tf.placeholder(dtype=tf.float32, shape=self.piece_2_out.shape)
                    self.piece_2_loss = tf.losses.mean_squared_error(self.piece_2_target, self.piece_2_out)
                else:
                    self.piece_2_out = up8

            # The third piece of the piecewise distillation
            with tf.variable_scope("piece2"):
                conv8 = slim.conv2d(self.piece_2_out, start_channel_depth * 2, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv8_1')
                conv8 = slim.conv2d(conv8, start_channel_depth * 2, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv8_2')

                up9 = upsample_and_concat(conv8, conv1, start_channel_depth, start_channel_depth * 2)
                conv9 = slim.conv2d(up9, start_channel_depth, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv9_1')
                conv9 = slim.conv2d(conv9, start_channel_depth, [3, 3], rate=1, activation_fn=tf.nn.relu, scope='g_conv9_2')

                conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')

                self.output = tf.depth_to_space(conv10, 2)

            # The loss, optimizer, and training op
            self.loss = tf.reduce_mean(tf.abs(self.output - self.labels))
            global_step = tf.Variable(0, trainable=False)
            # Add option for adjusting learning rate as in the paper
            self.learning_rate = tf.placeholder(tf.float32)

            # Optimizers and training ops
            # The optimizer for the full network
            full_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # The optimizer for the first piece
            first_piece_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "piece0")
            first_piece_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # The optimzier for the second piece
            second_piece_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "piece1")
            second_piece_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # The optimizer for the third piece
            third_piece_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "piece2")
            third_piece_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            # The training operations
            self.train_op_full = full_optimizer.minimize(self.loss, global_step=global_step)
            if self.student:
                first_piece_train = first_piece_optimizer.minimize(self.piece_1_loss)
                second_piece_train = second_piece_optimizer.minimize(self.piece_2_loss)
                third_piece_train = third_piece_optimizer.minimize(self.loss)

            # The lists that will be indexed into in the train ops
            if self.student:
                self.train_ops = [first_piece_train, second_piece_train, third_piece_train]
                self.losses = [self.piece_1_loss, self.piece_2_loss, self.loss]
                self.targets = [self.piece_1_target, self.piece_2_target, self.labels]

            # Create save operation
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.saver = tf.train.Saver(var_list=vars)
            # Create session and build parameters
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())


    def train_step(self, x, y, sess, learning_rate=None, piece=None):
        """
        Takes a training step on the batch fed in using the session given
        :param x: The input batch
        :param y: The output batch
        :param sess: The session to run this in
        :return: The value of the loss for this training step
        """
        assert piece >= 0 and piece < 3, "Input a valid piece of the network"

        if learning_rate is None:
            # Then pass in the default learning_rate
            learning_rate = self.start_learning_rate

        # Condition training on which piece we're currently training
        if piece is None:
            optim_step = self.train_op_full
            loss = self.loss
            target = self.labels
        else:
            optim_step = self.train_ops[piece]
            loss = self.losses[piece]
            target = self.targets[piece]

        feed_dict = {
            self.input: x,
            target: y,
            self.learning_rate: learning_rate
        }

        loss_value, _ = sess.run((loss, optim_step), feed_dict=feed_dict)

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

    def save_model(self, save_name=None):
        """
        Saves the model in the checkpoints folder
        :param save_name: The name under which to save the model
        :return: None
        """
        print("Saving model...")
        if save_name is not None:
            self.saver.save(self.sess, "./checkpoints/UNet" + save_name)
            return

        self.saver.save(self.sess, "./checkpoints/UNet" + str(self.start_channel_depth))

    def load_model(self, starting_depth):
        """
        Loads in the pre-trained weights from the specified model
        :param starting_depth: Specifies a model to load by the starting channel depth
        :return: None
        """
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        ckpt = tf.train.get_checkpoint_state("./checkpoints/")
        if ckpt:
            print('loaded ' + ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('load failed')
            exit(0)

    def rescope(self):
        """
        This method re-scopes the original UNet checkpoints to fit into our piecewise distillation scheme
        :return: None
        """
        reader = pywrap_tensorflow.NewCheckpointReader("./checkpoints/model.ckpt")
        var_to_shape = reader.get_variable_to_shape_map()
        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        unassigned = []

        count = 0
        for key in var_to_shape:
            assigned = False
            # Find the matching variable in traininable scope
            for trainable_var in trainables:
                if key in trainable_var.name:
                    # Increment the count
                    count += 1
                    #print(f"Assigning value to tensor {trainable_var.name}")
                    # Assing the value of this saved weight to the current graph
                    try:
                        tf.assign(trainable_var, reader.get_tensor(key))
                        trainables.remove(trainable_var)
                        assigned = True
                    except ValueError:
                        print(f"Shape mismatch rejection, skipping {trainable_var.name} for input tensor {key}")
                        count -= 1

            if not assigned and "Adam" not in key and "Variable" in key:
                unassigned += [key]
                print(f"Failed to assign tensor: {key}, size: {reader.get_tensor(key).shape}")
        try:
            tf.assign(trainables[0], reader.get_tensor(unassigned[0]))
            count += 1
            trainables.remove(trainables[0])
        except ValueError:
            print("fuck")
        print(f"Assigned {count} variables, failed to assign {trainables}")