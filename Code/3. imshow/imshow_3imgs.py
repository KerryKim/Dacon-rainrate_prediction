import numpy as np
import matplotlib.pyplot as plt



color_map = plt.cm.get_cmap('RdBu')
color_map = color_map.reversed()

image_sample = np.load('data/temp_test_fold/subset_029858_01.npy_2.npy')

plt.style.use('fivethirtyeight')
plt.figure(figsize=(30, 30))

for i in range(3):
    plt.subplot(1,4,i+1)
    plt.imshow(image_sample[:, :, i], cmap=color_map)

plt.subplot(1,4,4)
plt.imshow(image_sample[:,:,-1], cmap = color_map)
plt.show()
