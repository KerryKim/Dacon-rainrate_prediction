import os
import glob
import numpy as np


train_files = glob.glob('data/train/*.npy')

train_max_list = []
train_min_list = []
target_max_list = []
target_min_list = []


for file in train_files:
    train_data = np.load(file)

    feature = (train_data[:, :, :9])
    train_max_list.append(feature.max())
    train_min_list.append(feature.min())

    target = (train_data[:, :, -1])
    target_max_list.append(target.max())
    target_min_list.append(target.min())



print((np.array(train_max_list)).shape)
print((np.array(train_min_list)).shape)
print((np.array(target_max_list).shape))
print((np.array(target_min_list).shape))


print(max(train_max_list))
print(min(train_min_list))
print(max(target_max_list))
print(min(target_min_list))




