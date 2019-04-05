from timeit import Timer
from scipy import signal
import scipy
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow as tf

'''
TODOs:
tf convolutions
    implement function
    verify it's right by comparing to naive approach
cython convolutions
    implement
    verify correctness
Multiple Convolutions
    make all functions take in many filters (our starting model uses 2000 or more I believe)
        this will make methods like the tf one look better, because they have some startup time that gets amortized when
        your model grows
Model importing/conversion
    instead of building the tf model, lets be able to import it from disk (will help with testing our models)
    should also be able to convert model to numpy/scipy so we can test with things that aren't just tf
        make this import code run once, dont have it run on every forward pass

'''
model_dir = "./checkpoint/Sony/model.ckpt.meta"
checkpoint_dir = "./checkpoint"
# [batch, in_height, in_width, in_channels]
test_id = int("10045")
def tf_conv():

    with tf.Session() as sess:
        in_image = tf.placeholder(tf.float32, [None, None, None, 4])
        # gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
        out_image = forward(in_image)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(model_dir)
        saver.restore(sess, "./checkpoint/Sony/model.ckpt")

        in_path = "./images/10045_00_0.1s.ARW"
        gt_path = "./images/10045_00_0.04s.ARW"
        result_dir = "./results/"
        in_fn = os.path.basename(in_path)
        print(in_fn)
        gt_fn = os.path.basename(gt_path)
        print(gt_fn)
        in_exposure = .1
        gt_exposure = .04
        ratio = min(gt_exposure / in_exposure, 300)
        print("ratio is: ", ratio)

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) # * ratio
        Image.fromarray(raw.raw_image_visible.astype(np.float32), 'RGB').save(
            result_dir + 'final/%5d_00_%d_out.png' % (99999999, ratio))
        # im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
        #
        # gt_raw = rawpy.imread(gt_path)
        # im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        input_full = np.minimum(input_full, 1.0)

        output = sess.run(out_image, feed_dict={in_image: input_full})
        output = np.minimum(np.maximum(output, 0), 1)
        output = output[0, :, :, :]
        # gt_full = gt_full[0, :, :, :]
        # scale_full = scale_full[0, :, :, :]
        # scale_full = scale_full * np.mean(gt_full) / np.mean(
        #     scale_full)  # scale the low-light image to the same mean of the groundtruth

def tf_conv(input, filter):
    sess = tf.Session()
    with sess.as_default():
        # Convert function args into tensors of appropriate dimensions
        # input should be a 4D tensor of shape [batch_size, input shape, in_channels]
        # filter should be a 4D tensor of shape [filter shape, in_channels, out_channels]
        input = tf.convert_to_tensor(input)
        input = tf.expand_dims(input, axis=0)
        input = tf.expand_dims(input, axis=3)
        filter = tf.expand_dims(filter, axis=2)
        filter = tf.expand_dims(filter, axis=3)

        # Filter must be of type tf.float64
        filter = tf.cast(filter, tf.float64)

        # Convolve using VALID padding
        out = tf.nn.convolution(input, filter, padding="VALID")

        # out is a 4D tensor of shape [batch_size, output shape, out_channels]
        # so we need to extract the output
        out = tf.squeeze(out, axis=3)
        out = tf.squeeze(out, axis=0)
        out = out.eval()
        sess.close()
    return out

# def cython_conv(input, filter):
#     TODO

def scipy_conv(input, filter):
    out = signal.convolve(input, filter, mode='valid')
    return out

def naive_conv(input, filter):
    size_i = len(input)
    size_j = len(input[0])
    filter_width = len(filter)
    output = [[0 for _ in range(size_j-2)] for _ in range(size_i-2)]
    for i in range(size_i-2):
        for j in range(size_j-2):
            out_val = output[i][j]
            for fi in range(filter_width):
                for fj in range(filter_width):
                    out_val += filter[fi][fj] * input[i+fi][j+fj]
            output[i][j] = out_val
    return output


if __name__ == '__main__':
    test_mat = np.random.normal(size=(20, 10))
    # This is a gaussian blur kernel, I'm using it here because its relatively simple and because no components of it
    # are zero, which may (I dont know) result in misleading timeings)
    gaussian_blur = [[1 / 16, 2 / 16, 1 / 16],
                     [2 / 16, 4 / 16, 2 / 16],
                     [1 / 16, 2 / 16, 1 / 16]]

    assert(np.allclose(scipy_conv(test_mat, gaussian_blur), np.array(naive_conv(test_mat, gaussian_blur)), atol=1e-3))
    print("scipy and naive agreed")

    assert(np.allclose(tf_conv(test_mat, gaussian_blur), np.array(naive_conv(test_mat, gaussian_blur)), atol=1e-3))
    print("tensorflow and naive agreed")

    fps = lambda seconds_per_frame: round(1/seconds_per_frame, 5)

    # 720p HD
    mat = np.random.normal(size=(1080, 720))

    num_times = 1

    naive_timer = Timer(lambda: naive_conv(mat, gaussian_blur))
    scipy_timer = Timer(lambda: scipy_conv(mat, gaussian_blur))
    tf_timer = Timer(lambda: tf_conv(mat, gaussian_blur))
    print("naive fps: " + str(fps(naive_timer.timeit(number=num_times))))
    print("scipy fps: " + str(fps(scipy_timer.timeit(number=num_times))))
    print("tf fps: " + str(fps(tf_timer.timeit(number=num_times))))
