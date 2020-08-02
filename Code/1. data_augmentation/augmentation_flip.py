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

    x_flip_h = np.zeros_like(x_train)
    y_flip_h = np.zeros_like(x_target)

    for j in range(x_train.shape[2]):
        x_flip_h[:, :, j] = np.flip(x_train[:, :, j], 0)
    y_flip_h[:, :, 0] = np.flip(x_target[:, :, 0], 0)


    imgs_h = np.concatenate((x_flip_h, y_flip_h), axis=2)
    np.save('data/augmentation_flip/{}_flip_h'.format(os.path.basename(file)), imgs_h)

    x_flip_v = np.zeros_like(x_train)
    y_flip_v = np.zeros_like(x_target)

    for j in range(x_train.shape[2]):
        x_flip_v[:, :, j] = np.flip(x_train[:, :, j], 1)
    y_flip_v[:, :, 0] = np.flip(x_target[:, :, 0], 1)

    imgs_v = np.concatenate((x_flip_v, y_flip_v), axis=2)
    np.save('data/augmentation_flip/{}_flip_v'.format(os.path.basename(file)), imgs_v)