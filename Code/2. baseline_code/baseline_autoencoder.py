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


#Build residual block
def residual_block(input_ts):
    x = Conv2D(nfilters * 1, kernel_size=3, strides=1, padding='same')(input_ts)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(nfilters * 1, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    return Add()([x, input_ts])


#Define model
def build_model(input_layer, start_neurons):
    input_layer = BatchNormalization()(input_layer)

    # 40 x 40 -> 20 x 20
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(pool1)
    pool1 = Dropout(0.25)(pool1)

    # 20 x 20 -> 10 x 10
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(pool2)
    pool2 = Dropout(0.25)(pool2)

    # 10 x 10 -> 5 x 5
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(pool3)
    pool3 = Dropout(0.25)(pool3)

    # 5 x 5
    convm = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)

    # 5 x 5 -> 10 x 10
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.25)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = BatchNormalization()(uconv3)

    # 10 x 10 -> 20 x 20
    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.25)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = BatchNormalization()(uconv2)

    # 20 x 20 -> 40 x 40
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.25)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Dropout(0.25)(uconv1)
    output_layer = Conv2D(1, (1, 1), padding="same", activation='relu')(uconv1)

    return output_layer


#Define trainGenerator
def trainGenerator():
    train_path = temp_train_fold
    train_files = sorted(glob.glob(train_path + '/*'))

    for file in train_files:
        dataset = np.load(file)

        feature = dataset[:, :, :10]
        target = dataset[:, :, -1].reshape(40, 40, 1)


        cutoff_target_labels = np.where(target < 0, 0, target)
        cutoff_target_min_labels = target.min()

        '''
        if (cutoff_target_labels > 0).sum() <= 50:       # Pixel_rainrate_num < 50 is not of use
            continue
        if cutoff_target_min_labels < 0:             # Rainrate min for each data < 0 is not of use
            continue
        '''
        yield (feature, cutoff_target_labels)


#Define validGenerator
def validGenerator():
    valid_path = temp_valid_fold
    valid_files = sorted(glob.glob(valid_path + '/*'))

    for file in valid_files:
        dataset = np.load(file)

        feature = dataset[:, :, :10]
        target = dataset[:, :, -1].reshape(40, 40, 1)


        cutoff_target_labels = np.where(target < 0, 0, target)
        cutoff_target_min_labels = target.min()

        '''
        if (cutoff_target_labels > 0).sum() <= 50:       # Pixel_rainrate_num < 50 is not of use
            continue
        if cutoff_target_min_labels < 0:             # Rainrate min for each data < 0 is not of use
            continue
        '''

        yield (feature, cutoff_target_labels)


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
    n_split = 5
    test_nfolds = 1
    nfilters = 64
    epochs = 50
    batch_train = 32
    batch_valid = 128
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


    # Model
    input_layer = Input((40, 40, 10))
    output_layer = build_model(input_layer, nfilters)
    model = Model(input_layer, output_layer)
    model.summary()


    # Prepare train & valid data
    train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32),
                                                   (tf.TensorShape([40, 40, 10]), tf.TensorShape([40, 40, 1])))
    valid_dataset = tf.data.Dataset.from_generator(validGenerator, (tf.float32, tf.float32),
                                                   (tf.TensorShape([40, 40, 10]), tf.TensorShape([40, 40, 1])))
    train_dataset = train_dataset.batch(batch_train).prefetch(1)
    valid_dataset = valid_dataset.batch(batch_valid).prefetch(1)


    #5-fold cross evaluation
    for fold in range(nfolds):
        train_samples, valid_samples = generate_split()

        print("=====Train Model_{} iteration====".format(fold))
        check_path = 'result/checkpoints/weight_fold_{}.h5'.format(fold)
        callbacks = [EarlyStopping(monitor='val_score', patience=5, verbose=0),
                     ModelCheckpoint(check_path, monitor='val_loss', save_best_only=True, verbose=0)]
        model.compile(loss="mae", optimizer='adam', metrics=[score, fscore_k])
        model_history = model.fit(train_dataset, epochs=epochs, verbose=1, validation_data=valid_dataset, callbacks=callbacks)


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

