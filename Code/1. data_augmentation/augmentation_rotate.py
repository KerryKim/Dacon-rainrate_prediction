import random
import glob
import numpy as np
import os


#x_train, y_train(=x_target)
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

    #Rotate 90 degree
    rotate_X_90 = np.zeros_like(x_train)
    rotate_Y_90 = np.zeros_like(x_target)

    for i in range(10):
        rotate_X_90[:, :, i] = np.rot90(x_train[:, :, i])
    rotate_Y_90[:, :, 0] = np.rot90(x_target[:, :, 0])

    gen90_imgs = np.concatenate((rotate_X_90, rotate_Y_90), axis=2)
    np.save('data/augmentation_rotate/{}_rotate90.npy'.format(os.path.basename(file)), gen90_imgs)


    #Rotate 180 degree
    rotate_X_180 = np.zeros_like(x_train)
    rotate_Y_180 = np.zeros_like(x_target)

    for i in range(10):
        rotate_X_180[:, :, i] = np.rot90(x_train[:, :, i])
        rotate_X_180[:, :, i] = np.rot90(rotate_X_180[:, :, i])
    rotate_Y_180[:, :, 0] = np.rot90(x_target[:, :, 0])
    rotate_Y_180[:, :, 0] = np.rot90(rotate_Y_180[:, :, 0])

    gen180_imgs = np.concatenate((rotate_X_180, rotate_Y_180), axis=2)
    np.save('data/augmentation_rotate/{}_rotate180.npy'.format(os.path.basename(file)), gen180_imgs)


    #Rotate 270 degree
    rotate_X_270 = np.zeros_like(x_train)
    rotate_Y_270 = np.zeros_like(x_target)

    for i in range(10):
        rotate_X_270[:, :, i] = np.rot90(x_train[:, :, i])
        rotate_X_270[:, :, i] = np.rot90(rotate_X_270[:, :, i])
        rotate_X_270[:, :, i] = np.rot90(rotate_X_270[:, :, i])
    rotate_Y_270[:, :, 0] = np.rot90(x_target[:, :, 0])
    rotate_Y_270[:, :, 0] = np.rot90(rotate_Y_270[:, :, 0])
    rotate_Y_270[:, :, 0] = np.rot90(rotate_Y_270[:, :, 0])

    gen270_imgs = np.concatenate((rotate_X_270, rotate_Y_270), axis=2)
    np.save('data/augmentation_rotate/{}_rotate270.npy'.format(os.path.basename(file)), gen270_imgs)
