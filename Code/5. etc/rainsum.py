import numpy as np

dataset = np.load('data/train/subset_017873_05.npy')

feature= dataset[:, :, :3]
cutoff_labels_feature = np.where(feature <0, 0, feature)

print((cutoff_labels_feature>0).sum())

target = dataset[:, :, -1].reshape(40, 40, 1)
cutoff_labels_target = np.where(target < 0, 0, target)

print((cutoff_labels_target > 0).sum())


