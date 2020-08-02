import numpy as np
import matplotlib.pyplot as plt

color_map = plt.cm.get_cmap('RdBu')
color_map = color_map.reversed()

image_sample_0 = np.load('data/backup/train_origin/subset_010500_06.npy')
image_sample_1 = np.load('data/augmentation_cutmix/subset_010500_06.npy_cutmix.npy')
image_sample_2 = np.load('data/backup/train_origin/subset_014281_03.npy')
image_sample_3 = np.load('data/augmentation_cutmix/subset_014281_03.npy_cutmix.npy')
image_sample_4 = np.load('data/augmentation_cutmix/subset_014096_11.npy_cutmix.npy')


plt.style.use('fivethirtyeight')

plt.figure(figsize=(30, 30))

for i in range(10):
    plt.subplot(5,11,i+1)
    plt.imshow(image_sample_0[:, :, i], cmap=color_map)
    plt.subplot(5,11,i+12)
    plt.imshow(image_sample_1[:, :, i], cmap=color_map)
    plt.subplot(5,11,i+23)
    plt.imshow(image_sample_2[:, :, i], cmap=color_map)
    plt.subplot(5,11,i+34)
    plt.imshow(image_sample_3[:, :, i], cmap=color_map)
    plt.subplot(5,11,i+45)
    plt.imshow(image_sample_4[:, :, i], cmap=color_map)



plt.subplot(5,11,11)
plt.imshow(image_sample_0[:,:,-1], cmap = color_map)
plt.subplot(5,11,22)
plt.imshow(image_sample_1[:,:,-1], cmap = color_map)
plt.subplot(5,11,33)
plt.imshow(image_sample_2[:,:,-1], cmap = color_map)
plt.subplot(5,11,44)
plt.imshow(image_sample_3[:,:,-1], cmap = color_map)
plt.subplot(5,11,55)
plt.imshow(image_sample_4[:,:,-1], cmap = color_map)


plt.show()
