#Import module
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
import random

#Keras libraries
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Dropout, Conv2DTranspose, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#Etc
from sklearn.metrics import f1_score
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#Random seed
np.random.seed(7)
random.seed(7)
tf.random.set_seed(7)


#Define model
def get_model():
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_row_size, img_col_size, 3))
    for layer in base_model.layers[:15]:
        layer.trainable = False

    # 1 x 1 x 512 -> 2 x 2 x 512
    out = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(base_model.output)
    out = Dropout(0.5)(out)
    out = Conv2D(512, (3, 3), activation="relu", padding="same")(out)
    out = Conv2D(512, (3, 3), activation="relu", padding="same")(out)
    out = BatchNormalization()(out)

    # 2 x 2 x 512 -> 5 x 5 x 512
    out = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="valid")(out)
    out = Dropout(0.5)(out)
    out = Conv2D(512, (3, 3), activation="relu", padding="same")(out)
    out = Conv2D(512, (3, 3), activation="relu", padding="same")(out)
    out = BatchNormalization()(out)

    # 5 x 5 x 512 -> 10 x 10 x 256
    out = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(out)
    out = Dropout(0.5)(out)
    out = Conv2D(256, (3, 3), activation="relu", padding="same")(out)
    out = Conv2D(256, (3, 3), activation="relu", padding="same")(out)
    out = BatchNormalization()(out)

    # 10 x 10 x 256 -> 20 x 20 x 128
    out = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(out)
    out = Dropout(0.5)(out)
    out = Conv2D(128, (3, 3), activation="relu", padding="same")(out)
    out = BatchNormalization()(out)

    # 20 x 20 x 128 -> 40 x 40 x 64
    out = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(out)
    out = Dropout(0.5)(out)
    out = Conv2D(64, (3, 3), activation="relu", padding="same")(out)
    out = BatchNormalization()(out)

    # 40 x 40 x 64 -> 40 x 40 x 1
    output = Conv2D(1, (1, 1), padding="same", activation='relu')(out)

    model = Model(inputs=base_model.input, outputs=output)
    return model


#Make train data
def trainGenerator():
    train_path = 'data/temp_train_fold'
    train_files = sorted(glob.glob(train_path + '/*'))

    for file in train_files:
        dataset = np.load(file)
        target = dataset[:, :, -1].reshape(40, 40, 1)
        cutoff_labels = np.where(target < 0, 0, target)
        feature = dataset[:, :, :3]

        if (cutoff_labels > 0).sum() < 50:
            continue

        yield (feature, cutoff_labels)


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

def fscore_keras(y_true, y_pred):
    score = tf.py_function(func=fscore, inp=[y_true, y_pred], Tout=tf.float32, name='fscore_keras')
    return score

def maeOverFscore_keras(y_true, y_pred):
    score = tf.py_function(func=maeOverFscore, inp=[y_true, y_pred], Tout=tf.float32, name='custom_mse')
    return score


#Let get it started
if __name__ == "__main__":
    #Define parameters
    img_row_size, img_col_size = 40, 40

    #Prepare train data
    train_dataset = tf.data.Dataset.from_generator(trainGenerator, (tf.float32, tf.float32),
                                                   (tf.TensorShape([40, 40, 3]), tf.TensorShape([40, 40, 1])))
    train_dataset = train_dataset.batch(32).prefetch(1)

    #Make test data
    test_path = 'data/temp_test_fold'
    test_files = sorted(glob.glob(test_path + '/*'))
    X_test = []
    for file in tqdm(test_files, desc='test'):
        data = np.load(file)
        X_test.append(data[:, :, :3])
    X_test = np.array(X_test)
    print(X_test.shape)

    #Model Training
    model = get_model()
    model.summary()
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="mae", optimizer=sgd, metrics=[maeOverFscore_keras, fscore_keras])
    print('#Train Model')
    model_history = model.fit(train_dataset, epochs = 5, verbose=1)

    #Predict test data&Result
    pred = model.predict(X_test)
    print(pred.shape)

    submission = pd.read_csv('sample_submission.csv')
    submission.iloc[:,1:] = pred.reshape(-1, 1600)
    submission.to_csv('result/Dacon_baseline_transfer.csv', index = False)

