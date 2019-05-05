# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import numpy as np
import rawpy
import glob
from unet import UNet
from squeeze_UNet import Squeeze_UNet

"""
This is an adaptation of the code from https://bit.ly/2UAvptW
"""

input_dir = "../dataset/Sony/Sony/short/"
gt_dir = "../dataset/Sony/Sony/long/"
checkpoint_dir = '../result_Sony/'
result_dir = '../result_Sony/'

# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

assert train_fns != [], "train_fns is null, double check directory paths"
assert train_ids != [], "train_ids is null, double check directory paths"

validate_interval = 20
ps = 512  # patch size for training

using_GPU = True

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

def get_validation_loss(model):
    input_dir = "../dataset/Sony/Sony_val/short/"
    gt_dir = "../dataset/Sony/Sony_val/long/"

    assert (input_dir is not None) and (gt_dir is not None), "Set the variables above to the locations of the testing data ^^"

    test_fns = glob.glob(gt_dir + '2*.ARW')
    test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

    assert test_fns != [], "train_fns is null in get_val_loss, double check directory paths"
    assert test_ids != [], "train_ids is null in get_val_loss, double check directory paths"

    losses = []

    for test_id in test_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            in_fn = os.path.basename(in_path)
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
            scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            input_full = np.minimum(input_full, 1.0)

            loss = model.evaluate(input_full, gt_full, model.sess)

            # Add this to the list of losses
            losses += [loss]

    return np.mean(losses)


def main():
    # Raw data takes long time to load. Keep them in memory after loaded.
    gt_images = [None] * 6000
    input_images = {}
    input_images['300'] = [None] * len(train_ids)
    input_images['250'] = [None] * len(train_ids)
    input_images['100'] = [None] * len(train_ids)

    g_loss = np.zeros((5000, 1))

    allfolders = glob.glob('./result/*0')
    epochs = 500

    # Hyperparameters
    learning_rate = 1e-4
    # starting_channel_depths = [128, 64, 32, 16, 8, 4, 2, 1]
    starting_channel_depths = [64]

    # Keeps track of accuracies for different hyperparameters
    accuracies = []

    for starting_channel_depth in starting_channel_depths:
        # Build the model
        training_loss = 0
        if using_GPU:
            with tf.device('/device:GPU:1'):
                model = Squeeze_UNet(start_channel_depth=starting_channel_depth, learning_rate=learning_rate)
        else:
            model = Squeeze_UNet(start_channel_depth=starting_channel_depth, learning_rate=learning_rate)

        for epoch in range(epochs):
            print("training on epoch: {0}".format(epoch))
            cnt = 0
            if epoch > 2000:
                learning_rate = 1e-5

            for ind in np.random.permutation(len(train_ids)):
            # for ind in range(2):
                # get the path from image id
                train_id = train_ids[ind]
                in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
                in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
                in_fn = os.path.basename(in_path)

                gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
                gt_path = gt_files[0]
                gt_fn = os.path.basename(gt_path)
                in_exposure = float(in_fn[9:-5])
                gt_exposure = float(gt_fn[9:-5])
                ratio = min(gt_exposure / in_exposure, 300)

                st = time.time()
                cnt += 1

                if input_images[str(ratio)[0:3]][ind] is None:
                    raw = rawpy.imread(in_path)
                    input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

                    gt_raw = rawpy.imread(gt_path)
                    im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                    gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

                # crop
                H = input_images[str(ratio)[0:3]][ind].shape[1]
                W = input_images[str(ratio)[0:3]][ind].shape[2]

                xx = np.random.randint(0, W - ps)
                yy = np.random.randint(0, H - ps)
                input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]

                if using_GPU:
                    with tf.device('/device:GPU:1'):
                        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]
                else:
                    gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

                if np.random.randint(2, size=1)[0] == 1:  # random flip
                    input_patch = np.flip(input_patch, axis=1)
                    gt_patch = np.flip(gt_patch, axis=1)
                if np.random.randint(2, size=1)[0] == 1:
                    input_patch = np.flip(input_patch, axis=2)
                    gt_patch = np.flip(gt_patch, axis=2)
                if np.random.randint(2, size=1)[0] == 1:  # random transpose
                    input_patch = np.transpose(input_patch, (0, 2, 1, 3))
                    gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

                if using_GPU:
                    with tf.device('/device:GPU:1'):
                        input_patch = np.minimum(input_patch, 1.0)
                else:
                    input_patch = np.minimum(input_patch, 1.0)

                G_current = model.train_step(input_patch, gt_patch, model.sess)
                #output = np.minimum(np.maximum(output, 0), 1)
                g_loss[ind] = G_current
                training_loss = np.mean(g_loss[np.where(g_loss)])
                print("%d %d Loss=%.3f Time=%.3f" % (epoch, cnt, training_loss, time.time() - st))

            val_loss = get_validation_loss(model)
            model.save_model(epoch_index=epoch)
            with open("./losses.txt", "a+") as f:
                f.write("training loss is: {0} for epoch {1}\n".format(str(training_loss), str(epoch)))
                if (epoch % validate_interval == 0):
                    f.write("validation loss is: {0} for epoch {1}\n".format(str(val_loss), str(epoch)))

        accuracies += [[np.mean(g_loss[np.where(g_loss)]), get_validation_loss(model)]]

    # Save the accuracies as a numpy array
    print(accuracies)
    accuracies = np.array(accuracies)
    np.save("benchmark_results.npy", accuracies)

if __name__ == "__main__":
    main()
