import random
import glob
import numpy as np
import os

#size = image size, beta = beta distribution
def rand_box(size, beta):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - beta)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    ratio = abs(bbx1 - bbx2) * abs(bby1 - bby2) / (40 * 40)

    return bbx1, bby1, bbx2, bby2, ratio


gen_train, gen_target = [], []

#Image generator for augmentation
def cutmix_generator():

    train_files = glob.glob('data/train/*.npy')

    print((np.array(train_files)).shape)

    rand_origin = int(random.uniform(0, len(train_files)))
    rand_fetch = int(random.uniform(0, len(train_files)))

    train_origin_file = train_files[rand_origin]
    train_origin = np.load(train_origin_file)

    train_origin_feature = train_origin[:, :, :9]
    train_origin_target = train_origin[:, :, -1].reshape(40, 40, 1)

    train_fetch_file = train_files[rand_fetch]
    train_fetch = np.load(train_fetch_file)

    train_fetch_featrue = train_fetch[:, :, :9]
    train_fetch_target = train_fetch[:, :, -1].reshape(40, 40, 1)


    # beta(alpha,alpha) alpha = 1
    bbx1, bby1, bbx2, bby2, ratio = rand_box([40, 40], np.random.beta(1, 1))
    bbx1, bbx2, bby1, bby2 = min(bbx1, bbx2), max(bbx1, bbx2), min(bby1, bby2), max(bby1, bby2)


    train_origin_feature[bbx1:bbx2, bby1:bby2, :] = train_fetch_featrue[bbx1:bbx2, bby1:bby2, :]
    gen_feature = train_origin_feature

    gen_target = train_origin_target * (1 - ratio) + train_fetch_target * (ratio)

    print(gen_feature.shape)
    print(gen_target.shape)

    gen_imgs = np.concatenate((gen_feature, gen_target), axis=2)

    np.save('data/augmentation/{}_gen.npy'.format(os.path.basename(train_origin_file)), gen_imgs)



for i in range(1):
    cutmix_generator()
