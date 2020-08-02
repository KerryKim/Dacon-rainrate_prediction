import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def flatten(lst):
    result = []
    for item in lst:
        result.extend(item)
    return result

train_files = glob.glob('data/train/*.npy')

temp_list = []
land_list = []

for file in train_files:
    train_data = np.load(file)

    earth_type = train_data[:, :, 9]
    earth_type = np.where(earth_type//100 == 1 , 1, earth_type)
    earth_type = earth_type.tolist()

    earth_type_1 = flatten(earth_type)

    if earth_type_1.count(1) > 300:
        continue

    feature = train_data[:, :, :9]
    temp_list.append(feature.sum())


temp_list.sort()

plt.plot(temp_list)
plt.show()