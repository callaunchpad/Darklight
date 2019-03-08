from timeit import Timer
from scipy import signal
import numpy as np
import tensorflow as tf
from network import forward
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
model_dir = "/Users/David/Desktop/School/Launchpad/Darklight/src/model.ckpt.meta"
import os
cwd = os.getcwd()
print(cwd)
# [batch, in_height, in_width, in_channels]
def tf_conv(mat, gaussian_blur):
    mat = np.expand_dims(mat, axis=0)
    mat = np.expand_dims(mat, axis=3)
    input = tf.random_normal([1080, 720], mean=0, stddev=1)
    input = tf.expand_dims(input, 0)
    input = tf.expand_dims(input, 3)
    output = forward(input)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(model_dir)
        saver.restore(sess, "model.ckpt")

        out = sess.run(output, feed_dict={input : mat})
        print(tf.shape(out))
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
    print("functions agreed")
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
