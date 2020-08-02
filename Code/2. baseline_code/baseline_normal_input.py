#Import module
import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import shutil

#Keras libraries
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Conv2DTranspose, MaxPooling2D, BatchNormalization, Activation, concatenate, Input, Add
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#Etc
from shutil import copyfile
from sklearn.metrics import f1_score
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#Random seed
np.random.seed(7)
random.seed(7)
tf.random.set_seed(7)

#Clear temp train & valid fold
def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


#Flatten list
def flatten(lst):
    result = []
    for item in lst:
        result.extend(item)
    return result


#Define model
def build_model():
    #inputs=Input((40, 40 ,10), dtype='float')
    inputs = Input(x_train.shape[1:])

    bn = BatchNormalization()(inputs)
    conv0 = Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu')(bn)

    bn = BatchNormalization()(conv0)
    conv = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(bn)
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


#Define trainGenerator
def trainGenerator():
    train_path = temp_train_fold
    train_files = sorted(glob.glob(train_path + '/*'))

    train = []
    for file in train_files:
        try:
            data = np.load(file)
            train.append(data)
        except:
            continue

    train = np.array(train)
    x_train = train[:, :, :, :10]
    y_train = train[:, :, :, -1].reshape(-1, 40, 40, 1)         #target value에서는 채널 차원을 reshape로 추가필요

    return x_train, y_train



#Define validGenerator
def validGenerator():
    valid_path = temp_valid_fold
    valid_files = sorted(glob.glob(valid_path + '/*'))

    valid = []
    for file in valid_files:
        try:
            data = np.load(file)
            valid.append(data)
        except:
            continue

    valid = np.array(valid)
    x_valid = valid[:, :, :, :10]
    y_valid = valid[:, :, :, -1].reshape(-1, 40, 40, 1)


    return x_valid, y_valid


#Make train & valid samples
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


#Define evaluation method
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

def fscore_k(y_true, y_pred):
    score = tf.py_function(func=fscore, inp=[y_true, y_pred], Tout=tf.float32, name='fscore_keras')
    return score

def score(y_true, y_pred):
    score = tf.py_function(func=maeOverFscore, inp=[y_true, y_pred], Tout=tf.float32, name='custom_mse')
    return score


#Let get it started
if __name__ == "__main__":

    #Define parameters
    nfolds = 5              #if nfolds=1, code not work (temp_train_fold is all train data)
    n_split = 50
    test_nfolds = 1

    epochs = 50
    batch_train = 32
    batch_valid = 32
    ensemble = 0
    img_row_size, img_col_size = 40, 40

    temp_train_fold = 'data/temp_train_fold'
    temp_valid_fold = 'data/temp_valid_fold'


    #Make test data
    test_path = 'data/test'
    test_files = sorted(glob.glob(test_path + '/*'))
    X_test = []
    for file in tqdm(test_files, desc='test'):
        data = np.load(file)
        X_test.append(data[:, :, :10])
    X_test = np.array(X_test)

    train_samples, valid_samples = generate_split()

    x_train, y_train = trainGenerator()
    x_valid, y_valid = validGenerator()

    print(x_train.shape)
    print(y_train.shape)
    print(x_valid.shape)
    print(y_valid.shape)

    # Model
    model = build_model()
    model.summary()


    # Prepare train & valid data
    # train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32),
    #                                                (tf.TensorShape([40, 40, 10]), tf.TensorShape([40, 40, 1])))
    # valid_dataset = tf.data.Dataset.from_generator(validGenerator, (tf.float32, tf.float32),
    #                                                (tf.TensorShape([40, 40, 10]), tf.TensorShape([40, 40, 1])))
    # train_dataset = train_dataset.batch(batch_train).prefetch(1)
    # valid_dataset = valid_dataset.batch(batch_valid).prefetch(1)



    #5-fold cross evaluation
    for fold in range(nfolds):


        print("=====Train Model_{} iteration====".format(fold))
        check_path = 'result/checkpoints/weight_fold_{}.h5'.format(fold)
        callbacks = [ModelCheckpoint(check_path, monitor='val_loss', save_best_only=True, verbose=0)]
        model.compile(loss="mae", optimizer="adam", metrics=[score, fscore_k])
        model_history = model.fit(x_train, y_train, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid), callbacks=callbacks)


        #Predict test data
        for j in range(test_nfolds):
            pred = model.predict(X_test)
            if j == 0:
                result = pred
            else:
                result += pred
        result /= test_nfolds

        submission = pd.read_csv('sample_submission.csv')
        submission.iloc[:, 1:] = result.reshape(-1, 1600)
        submission.to_csv('result/baseline_master_200514_{}.csv'.format(fold), index=False)


    #Ensemble test_results
    for fold in range(nfolds):
        test_result = pd.read_csv('result/baseline_master_200514_{}.csv'.format(fold))
        test_result = pd.DataFrame(test_result).values
        ensemble += test_result[:, 1:] * 1. / nfolds

    submission = pd.read_csv('sample_submission.csv')
    submission.iloc[:, 1:] = ensemble.reshape(-1, 1600)
    submission.to_csv('result/rainrate_ens_200510.csv', index=False)

