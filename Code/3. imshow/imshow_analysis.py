import numpy as np
import matplotlib.pyplot as plt

color_map = plt.cm.get_cmap('RdBu')
color_map = color_map.reversed()

image_sample_0 = np.load('data/train/subset_010464_02.npy')
image_sample_1 = np.load('data/augmentation_flip/subset_010464_02.npy_flip_h.npy')
image_sample_2 = np.load('data/augmentation_flip/subset_010464_02.npy_flip_v.npy')
image_sample_3 = np.load('data/augmentation_rotate/subset_010464_02.npy_rotate90.npy')
image_sample_4 = np.load('data/augmentation_rotate/subset_010464_02.npy_rotate180.npy')

plt.style.use('fivethirtyeight')

plt.figure(figsize=(30, 30))

for i in range(9):
    plt.subplot(5,10,i+1)
    plt.imshow(image_sample_0[:, :, i], cmap=color_map)
    plt.subplot(5,10,i+11)
    plt.imshow(image_sample_1[:, :, i], cmap=color_map)
    plt.subplot(5,10,i+21)
    plt.imshow(image_sample_2[:, :, i], cmap=color_map)
    plt.subplot(5,10,i+31)
    plt.imshow(image_sample_3[:, :, i], cmap=color_map)
    plt.subplot(5,10,i+41)
    plt.imshow(image_sample_4[:, :, i], cmap=color_map)


plt.subplot(5,10,10)
plt.imshow(image_sample_0[:,:,-1], cmap = color_map)
plt.subplot(5,10,20)
plt.imshow(image_sample_1[:,:,-1], cmap = color_map)
plt.subplot(5,10,30)
plt.imshow(image_sample_2[:,:,-1], cmap = color_map)
plt.subplot(5,10,40)
plt.imshow(image_sample_3[:,:,-1], cmap = color_map)
plt.subplot(5,10,50)
plt.imshow(image_sample_4[:,:,-1], cmap = color_map)


plt.show()