import random
import glob
import numpy as np
import os

'''
path = 'data/test_normalization/subset_029889_04.npy_norm.npy'

data = np.load(path)
print(data.shape)

'''

train_files = glob.glob('data/train_valuable/*.npy')


temp_max_list = []
temp_min_list = []
earth_max_list = []
earth_min_list = []


for file in train_files:
    dataset = np.load(file)

    x_train_temp = dataset[:, :, :9]
    temp_max_list.append(x_train_temp.max())
    temp_min_list.append(x_train_temp.min())

    x_train_earth = dataset[:, :, -1]
    earth_max_list.append(x_train_earth.max())
    earth_min_list.append(x_train_earth.min())


x_temp_max = (np.array(temp_max_list)).max()
x_temp_min = (np.array(temp_min_list)).min()

x_earth_max = (np.array(earth_max_list)).max()
x_earth_min = (np.array(earth_min_list)).min()

print(x_temp_max)
print(x_temp_min)
print(x_earth_max)
print(x_earth_min)


train_files_2 = glob.glob('data/test/*.npy')

for file in train_files_2:
    dataset = np.load(file)

    x_train_temp = dataset[:, :, :9]
    x_train_temp_norm = (x_train_temp - x_temp_min) / (x_temp_max - x_temp_min)
    
    x_train_earth = dataset[:, :, 10]
    x_train_earth_norm = (x_train_earth - x_earth_min) / (x_earth_max - x_earth_min)
    x_train_earth_norm = x_train_earth_norm.reshape(40, 40, 1)
    
    x_target = dataset[:, :, -1].reshape(40, 40, 1)
    
    imgs = np.concatenate((x_train_temp_norm, x_train_earth_norm,x_target), axis=2)
    np.save('data/test_norm/{}_norm'.format(os.path.basename(file)), imgs)
