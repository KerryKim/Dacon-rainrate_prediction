import numpy as np

def flatten(lst):
    result = []
    for item in lst:
        result.extend(item)
    return result

dataset = np.load('data/train/subset_010624_03.npy')
earth_type = dataset[:, :, 9]

earth_type = np.where(earth_type//100 == 1 , 1, earth_type)
earth_type = earth_type.tolist()

print(earth_type)
earth_type_1 = flatten(earth_type)
print(earth_type_1)
print(earth_type_1.count(1))
