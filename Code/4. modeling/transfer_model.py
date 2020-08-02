from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, Activation, Input, \
    Conv2DTranspose, BatchNormalization

img_row_size, img_col_size = 224, 224

model = VGG16(include_top=False, weights='imagenet', input_shape=(img_row_size, img_col_size, 3))

# 7 x 7 x 512 -> 14 x 14 x 512
out = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(model.ouput)
out = Dropout(0.25)(out)
out = Conv2D(512, (3, 3), activation="relu", padding="same")(out)
out = Conv2D(512, (3, 3), activation="relu", padding="same")(out)
out = BatchNormalization()(out)

# 14 x 14 x 512 -> 28 x 28 x 512
out = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(model.ouput)
out = Dropout(0.25)(out)
out = Conv2D(512, (3, 3), activation="relu", padding="same")(out)
out = Conv2D(512, (3, 3), activation="relu", padding="same")(out)
out = BatchNormalization()(out)



out = Conv2D(1, (3, 3), activation="relu", padding="same")(model.output)
out = Conv2D(1, (3, 3), activation="relu", padding="same")(out)
out = Conv2D(1, (3, 3), activation="relu", padding="same")(out)

out = Dense(128, activation='relu')(out)
out = Dropout(0.5)(out)

out = Conv2D(1, (3, 3), activation="relu", padding="same")(model.output)
out = Conv2D(1, (3, 3), activation="relu", padding="same")(out)
out = Conv2D(1, (3, 3), activation="relu", padding="same")(out)

out = Dense(64, activation='relu')(out)
out = Dropout(0.5)(out)

out = Conv2D(1, (3, 3), activation="relu", padding="same")(model.output)
out = Conv2D(1, (3, 3), activation="relu", padding="same")(out)
out = Conv2D(1, (3, 3), activation="relu", padding="same")(out)

out = Dense(32, activation='relu')(out)
out = Dropout(0.5)(out)

model.summary()