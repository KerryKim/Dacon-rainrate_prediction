import random
import glob
import numpy as np
import os

train_files = glob.glob('data/augmentation_cutmix/*.npy')

for file in train_files:
    dataset = np.load(file)

    x_train = dataset[:, :, :10]
    x_target = dataset[:, :, -1].reshape(40, 40, 1)

    cutoff_target_labels = np.where(x_target < 0, 0, x_target)
    cutoff_target_min_labels = x_target.min()

    if (cutoff_target_labels > 0).sum() < 50:  # Pixel_rainrate_num < 50 is not of use
        continue
    if cutoff_target_min_labels < 0:  # Rainrate min for each data < 0 is not of use
        continue


    imgs = np.concatenate((x_train, x_target), axis=2)
    np.save('data/augmentation_cutmix_valuable/{}_2'.format(os.path.basename(file)), imgs)

