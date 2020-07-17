import os
import glob
import pandas as pd
import numpy as np
import random
import shutil
from shutil import copyfile

def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def generate_split():

    #Make folders to save K fold data set(train/valid)
    def generate_temp_folder(path):
        clear_dir(path)

    generate_temp_folder(temp_train_fold)
    generate_temp_folder(temp_valid_fold)

    train_samples = 0
    valid_samples = 0

    files = glob.glob('data/train/*.npy')
    for fl in files:
        if np.random.randint(n_split)!= 1:
            copyfile(fl, 'data/temp_train_fold/{}'.format(os.path.basename(fl)))
            train_samples += 1
        else:
            copyfile(fl, 'data/temp_valid_fold/{}'.format(os.path.basename(fl)))
            valid_samples += 1

    print('# {} train samples | {} valid samples'.format(train_samples, valid_samples))
    print()
    return train_samples, valid_samples

n_split=20
temp_train_fold = 'data/temp_train_fold'
temp_valid_fold = 'data/temp_valid_fold'

train_samples, valid_samples = generate_split()