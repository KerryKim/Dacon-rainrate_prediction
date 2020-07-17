import random
import glob
import numpy as np
import os

train_files = glob.glob('data/backup/train_origin/*.npy')

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

    x_T = np.zeros_like(x_train)
    y_T = np.zeros_like(x_target)

    for j in range(10):
        x_T[:, :, j] = x_train[:, :, j].T

    y_T[:, :, 0] = x_target[:, :, 0].T


    imgs = np.concatenate((x_T, y_T), axis=2)
    np.save('data/augmentation_transpose/{}_transpose'.format(os.path.basename(file)), imgs)