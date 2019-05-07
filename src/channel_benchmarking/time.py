from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import numpy as np
import rawpy
import glob
from unet import UNet
import matplotlib.pyplot as plt

starting_channel_depths = [64, 32, 16, 8, 4, 2, 1]
in_path = "/Users/zacharylieberman/desktop/20005_01_0.04s.ARW"

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

in_fn = os.path.basename(in_path)
gt_files = glob.glob(in_path)
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

for depth in [16]:
    model = UNet(start_channel_depth=depth)
    model.load_model(depth)
    start_time = time.time()
    output = model.predict(input_full, model.sess)
    end_time = time.time()
    print(end_time - start_time)
    plt.imshow(np.squeeze(output.astype(int)))
    plt.show()
