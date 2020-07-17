import numpy as np
import pandas as pd
import os
#import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, Add, BatchNormalization, concatenate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

submission = pd.read_csv('sample_submission.csv')
train_files = os.listdir('data/backup/train_origin')

train = []
for file in train_files:
    try:
        data = np.load('data/train/' + file).astype('float32')
        train.append(data)
    except:
        continue

test = []
for sub_id in submission['id']:
    data = np.load('data/test/' + 'subset_' + sub_id + '.npy').astype('float32')
    test.append(data)

train = np.array(train)
test = np.array(test)

x_train = train[:,:,:,:10]
y_train = train[:,:,:,14]
test = test[:,:,:,:10]

del train

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.025, random_state=7777)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

y_train_ = y_train.reshape(-1,y_train.shape[1]*y_train.shape[2])

x_train = np.delete(x_train, np.where(y_train_<0)[0], axis=0)
y_train = np.delete(y_train, np.where(y_train_<0)[0], axis=0)
y_train = y_train.reshape(-1, x_train.shape[1], x_train.shape[2],1)
y_test = y_test.reshape(-1, y_test.shape[1], y_test.shape[2],1)

y_train_ = np.delete(y_train_, np.where(y_train_<0)[0], axis=0)

print(x_train.shape, y_train.shape)

x_train = x_train[np.sum((y_train_>= 50))]
y_train = y_train[np.sum((y_train_>= 50))]

print(x_train.shape, y_train.shape)


def mae_over_fscore(y_true, y_pred):
    '''
    y_true: sample_submission.csv 형태의 실제 값
    y_pred: sample_submission.csv 형태의 예측 값
    '''

    y_true = np.array(y_true)
    y_true = y_true.reshape(1, -1)[0]

    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(1, -1)[0]

    # 실제값이 0.1 이상인 픽셀의 위치 확인
    IsGreaterThanEqualTo_PointOne = y_true >= 0.1

    # 실제 값에 결측값이 없는 픽셀의 위치 확인
    IsNotMissing = y_true >= 0

    # mae 계산
    mae = np.mean(np.abs(y_true[IsGreaterThanEqualTo_PointOne] - y_pred[IsGreaterThanEqualTo_PointOne]))

    # f1_score 계산 위해, 실제값에 결측값이 없는 픽셀에 대해 1과 0으로 값 변환
    y_true = np.where(y_true[IsNotMissing] >= 0.1, 1, 0)

    y_pred = np.where(y_pred[IsNotMissing] >= 0.1, 1, 0)

    # f1_score 계산
    f_score = f1_score(y_true, y_pred)
    # f1_score가 0일 나올 경우를 대비하여 소량의 값 (1e-07) 추가
    return mae / (f_score + 1e-07)


def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true = y_true.reshape(1, -1)[0]

    y_pred = y_pred.reshape(1, -1)[0]

    over_threshold = y_true >= 0.1

    return np.mean(np.abs(y_true[over_threshold] - y_pred[over_threshold]))


def fscore(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    y_true = y_true.reshape(1, -1)[0]

    y_pred = y_pred.reshape(1, -1)[0]

    remove_NAs = y_true >= 0

    y_true = np.where(y_true[remove_NAs] >= 0.1, 1, 0)

    y_pred = np.where(y_pred[remove_NAs] >= 0.1, 1, 0)

    return (f1_score(y_true, y_pred))


def maeOverFscore(y_true, y_pred):
    return mae(y_true, y_pred) / (fscore(y_true, y_pred) + 1e-07)


def fscore_keras(y_true, y_pred):
    score = tf.py_function(func=fscore, inp=[y_true, y_pred], Tout=tf.float32, name='fscore_keras')
    return score


def score(y_true, y_pred):
    score = tf.py_function(func=maeOverFscore, inp=[y_true, y_pred], Tout=tf.float32, name='custom_mse')
    return score


def create_model():
    inputs = Input(x_train.shape[1:])

    bn = BatchNormalization()(inputs)
    conv0 = Conv2D(256, kernel_size=1, strides=1, padding='same', activation='relu')(bn)

    bn = BatchNormalization()(conv0)
    conv = Conv2D(128, kernel_size=2, strides=1, padding='same', activation='relu')(bn)
    concat = concatenate([conv0, conv], axis=3)

    bn = BatchNormalization()(concat)
    conv = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(bn)
    concat = concatenate([concat, conv], axis=3)

    for i in range(5):
        bn = BatchNormalization()(concat)
        conv = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(bn)
        concat = concatenate([concat, conv], axis=3)

    bn = BatchNormalization()(concat)
    outputs = Conv2D(1, kernel_size=1, strides=1, padding='same', activation='relu')(bn)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def train_model(x_data, y_data, k, s):
    k_fold = KFold(n_splits=k, shuffle=True, random_state=7777)

    model_number = 0
    for train_idx, val_idx in k_fold.split(x_data):
        if model_number == s:
            x_train, y_train = x_data[train_idx], y_data[train_idx]
            x_val, y_val = x_data[val_idx], y_data[val_idx]

            # 데이터를 부풀릴시 많은 양의 메모리가 필요
            x_train, y_train = data_generator(x_train, y_train)

            model = create_model()

            model.compile(loss='mae', optimizer='adam', metrics=[score, fscore_keras])

            callbacks_list = [
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    patience=3,
                    factor=0.8
                ),

                tf.keras.callbacks.ModelCheckpoint(
                    filepath='models/model' + str(model_number) + '.h5',
                    monitor='val_score',
                    save_best_only=True
                )
            ]

            model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_val, y_val),
                      callbacks=callbacks_list)

        model_number += 1

k = 5
models = []

train_model(x_train, y_train, k=k, s=4)

for n in range(k):
    model = load_model('models/model'+str(n)+'.h5', custom_objects = {'score':score,'fscore_keras':fscore_keras})
    models.append(model)