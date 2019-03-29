import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import rawpy
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout
from keras.layers import concatenate, Conv2DTranspose, BatchNormalization
from keras import backend as K

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def lrelu(x):
    return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

def forward(input):
    with tf.device('/gpu:0'):
        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
        conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
        conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
        conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
        conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

        up6 = upsample_and_concat(conv5, conv4, 256, 512)
        conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
        conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

        up7 = upsample_and_concat(conv6, conv3, 128, 256)
        conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
        conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

        up8 = upsample_and_concat(conv7, conv2, 64, 128)
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
        conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

        up9 = upsample_and_concat(conv8, conv1, 32, 64)
        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
        conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

        conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
        out = tf.depth_to_space(conv10, 2)
    return out

def fire_module(input, squeeze_output, expand_output, scopes, name=None):
    """channel axis is 3 i think for 4d"""
    #conv2d with 1x1
    #batch_normal
    input = tf.identity(input, name=name) # for debugging
    input = slim.conv2d(input, squeeze_output, [1, 1], rate=1, activation_fn=lrelu, padding='same', scope=scopes[0])
    input = slim.batch_norm(input)

    #conv2d with 1x1
    #conv2d with 3x3
    left = slim.conv2d(input, expand_output // 2, [1, 1], rate=1, activation_fn=lrelu, padding='same', scope=scopes[1])
    right = slim.conv2d(input, expand_output - expand_output // 2, [3, 3], rate=1, activation_fn=lrelu, padding='same', scope=scopes[2])
    #concat
    result = tf.concat([left, right], axis=-1)
    return result

def squeezeUNet(input):

    conv1 = fire_module(input, 16, 64, scopes=['g_conv1_fm1_squeeze', 'g_conv1_fm1_left', 'g_conv1_fm1_right'], name="input")
    print(conv1.get_shape())
    conv1 = fire_module(input, 16, 64, scopes=['g_conv1_fm2_squeeze', 'g_conv1_fm2_left', 'g_conv1_fm2_right'])
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')
    print(conv1.get_shape())
    print(pool1.get_shape())
    print(tf.shape(pool1))

    conv2 = fire_module(input, 32, 128, scopes=['g_conv2_fm1_squeeze', 'g_conv2_fm1_left', 'g_conv2_fm1_right'])
    conv2 = fire_module(input, 32, 128, scopes=['g_conv2_fm2_squeeze', 'g_conv2_fm2_left', 'g_conv2_fm2_right'])
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')
    print(tf.shape(conv2))
    print(tf.shape(pool2))

    conv3 = fire_module(input, 48, 192, scopes=['g_conv3_fm1_squeeze', 'g_conv3_fm1_left', 'g_conv3_fm1_right'])
    conv3 = fire_module(input, 48, 192, scopes=['g_conv3_fm2_squeeze', 'g_conv3_fm2_left', 'g_conv3_fm2_right'])
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')
    print(tf.shape(conv3))
    print(tf.shape(pool3))

    conv4 = fire_module(input, 64, 256, scopes=['g_conv4_fm1_squeeze', 'g_conv4_fm1_left', 'g_conv4_fm1_right'])
    conv4 = fire_module(input, 64, 256, scopes=['g_conv4_fm2_squeeze', 'g_conv4_fm2_left', 'g_conv4_fm2_right'])
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')
    print(tf.shape(conv4))
    print(tf.shape(pool4))

    conv5 = fire_module(input, 80, 320, scopes=['g_conv5_fm1_squeeze', 'g_conv5_fm1_left', 'g_conv5_fm1_right'])
    conv5 = fire_module(input, 80, 320, scopes=['g_conv5_fm2_squeeze', 'g_conv5_fm2_left', 'g_conv5_fm2_right'])
    print(tf.shape(conv5))

    # out chan in chan
    up6 = upsample_and_concat(conv5, conv4, 256, 320)
    conv6 = fire_module(input, 64, 256, scopes=['g_conv6_fm1_squeeze', 'g_conv6_fm1_left', 'g_conv6_fm1_right'])
    conv6 = fire_module(input, 64, 256, scopes=['g_conv6_fm2_squeeze', 'g_conv6_fm2_left', 'g_conv6_fm2_right'])

    up7 = upsample_and_concat(conv6, conv3, 192, 256)
    conv7 = fire_module(input, 48, 192, scopes=['g_conv7_fm1_squeeze', 'g_conv7_fm1_left', 'g_conv7_fm1_right'])
    conv7 = fire_module(input, 48, 192, scopes=['g_conv7_fm2_squeeze', 'g_conv7_fm2_left', 'g_conv7_fm2_right'])

    up8 = upsample_and_concat(conv7, conv2, 128, 192)
    conv8 = fire_module(input, 32, 128, scopes=['g_conv8_fm1_squeeze', 'g_conv8_fm1_left', 'g_conv8_fm1_right'])
    conv8 = fire_module(input, 32, 128, scopes=['g_conv8_fm2_squeeze', 'g_conv8_fm2_left', 'g_conv8_fm2_right'])

    up9 = upsample_and_concat(conv8, conv1, 64, 128)
    conv9 = fire_module(input, 16, 64, scopes=['g_conv9_fm1_squeeze', 'g_conv9_fm1_left', 'g_conv9_fm1_right'])
    conv9 = fire_module(input, 16, 64, scopes=['g_conv9_fm2_squeeze', 'g_conv9_fm2_left', 'g_conv9_fm2_right'])

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10, 2, name="network_output")
    return out
