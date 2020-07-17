import numpy as np
import matplotlib.pyplot as plt



color_map = plt.cm.get_cmap('RdBu')
color_map = color_map.reversed()

image_sample = np.load('data/augmentation_transpose/subset_010463_04.npy_transpose.npy')

plt.style.use('fivethirtyeight')

plt.figure(figsize=(30, 30))

for i in range(10):
    plt.subplot(1,11,i+1)
    plt.imshow(image_sample[:, :, i], cmap=color_map)

plt.subplot(1,11,11)
plt.imshow(image_sample[:,:,-1], cmap = color_map)
plt.show()