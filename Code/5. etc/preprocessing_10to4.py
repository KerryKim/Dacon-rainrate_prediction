import os
import glob
import numpy as np
import shutil
from shutil import rmtree

#Make temp train data & valid data
def generate_temp_folder(path):
    if os.path.exists(path):
        rmtree(path)
    os.mkdir(path)

temp_train_fold = 'data/temp_train_fold'
temp_test_fold = 'data/temp_test_fold'

generate_temp_folder(temp_train_fold)
generate_temp_folder(temp_test_fold)


def train_data_maker():
    train_files = glob.glob('data/train/*.npy')

    for file in train_files:
        train_data = np.load(file)
        train_target = train_data[:, :, -1].reshape(40, 40, 1)

        for j in range(3):
            feature = (train_data[:, :, 3*j:3*j+3])
            p_train_data = np.concatenate((feature, train_target), axis=2)

            np.save('data/temp_train_fold/{}_{}.npy'.format(os.path.basename(file),j), p_train_data)


def test_data_maker():
    test_files = glob.glob('data/test/*.npy')

    for file in test_files:
        test_data = np.load(file)

        for j in range(3):
            p_test_data = (test_data[:, :, 3*j:3*j+3])
            np.save('data/temp_test_fold/{}_{}.npy'.format(os.path.basename(file), j), p_test_data)


train_data_maker()
test_data_maker()



