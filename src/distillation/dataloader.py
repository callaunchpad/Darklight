import numpy as np
import rawpy
import glob
import os
import itertools

class DataLoader:

    def __init__(self, input_dir, ground_truth_dir, batch_size, is_validation=False):
        """
        Initializes the dataloader
        :param input_dir: The location of the input images
        :param ground_truth_dir: The location of the ground truth images
        :param batch_size: The batch size we wish to pull
        """
        self.input_dir = input_dir
        self.ground_truth_dir = ground_truth_dir
        self.batch_size = batch_size
        self.is_valid = is_validation
        self.batch_builder = self.batch_generator()

    def pack_raw(self, raw):
        """
        Converts a raw Bayer input to a numpy array
        :param raw: The raw image input
        :return: A numpy array that we can pass into the network
        """
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

    def load_image(self, input_path, ground_truth_path):
        """
        Returns the image as a numpy array taken from relative path.
        Much of the code in this function is taken from https://bit.ly/2UAvptW
        :param relative_path: The relative path to the image
        :return: The image as a numpy array
        """
        # The size of patches to sample
        patch_size = 512

        # The full path as far as the OS is concerned
        full_input_path = os.path.basename(input_path)
        full_ground_truth_path = os.path.basename(ground_truth_path)

        # Exposures on the images
        in_exposure = float(full_input_path[9:-5])
        gt_exposure = float(full_ground_truth_path[9:-5])

        # The exposure ratio between input and ground truth
        ratio = min(gt_exposure / in_exposure, 300)

        # Load and process the raw images
        input_raw = rawpy.imread(self.input_dir + full_input_path)
        input_full = np.expand_dims(self.pack_raw(input_raw), axis=0) * ratio

        gt_raw = rawpy.imread(self.ground_truth_dir + full_ground_truth_path)
        gt_full = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(gt_full / 65535.0), axis=0)

        # Random crop to the image
        H = input_full.shape[1]
        W = input_full.shape[2]

        # Grab a random patch from the image
        xx = np.random.randint(0, W - patch_size)
        yy = np.random.randint(0, H - patch_size)
        input_patch = input_full[:, yy:yy + patch_size, xx:xx + patch_size, :]
        gt_patch = gt_full[:, yy * 2:yy * 2 + patch_size * 2, xx * 2:xx * 2 + patch_size * 2, :]

        # Add in random flips and transposes to the dataset
        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        # This scales the image to the correct input domain
        input_patch = np.minimum(input_patch, 1.0)

        return (input_patch, gt_patch)

    def batch_generator(self):
        # Get a random permutation of the input files
        train_fns = glob.glob(self.ground_truth_dir + '0*.ARW')
        train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]

        if self.is_valid:
            iterator_cycle = itertools.repeat(np.random.permutation(train_ids), 1)
        else:
            iterator_cycle = itertools.cycle(np.random.permutation(train_ids))

        while True:
            input_batch, gt_batch = [], []

            for _ in range(self.batch_size):
                # Grab the current file
                next_id = next(iterator_cycle)

                input_files = glob.glob(self.input_dir + '%05d*.ARW' % next_id)
                in_path = input_files[np.random.random_integers(0, len(input_files) - 1)]

                gt_files = glob.glob(self.ground_truth_dir + '%05d_00*.ARW' % next_id)
                gt_path = gt_files[0]

                # Pull the images
                input_image, ground_truth_image = self.load_image(in_path, gt_path)

                # Add to batch
                input_batch += [np.squeeze(input_image)]
                gt_batch += [np.squeeze(ground_truth_image)]

            yield np.array(input_batch), np.array(gt_batch)

    def get_next_batch(self):
        return next(self.batch_builder)

