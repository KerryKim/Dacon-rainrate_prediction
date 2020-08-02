from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, Activation, Input, \
    Conv2DTranspose, BatchNormalization

img_row_size, img_col_size = 40, 40


def get_model():
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(img_row_size, img_col_size, 3))
    base_encoder = Model(inputs=base_model.input, outputs=base_model.get_layer("block3_pool").output)
    for layer in base_model.layers[:9]:
        layer.trainable = False

    # 5 x 5 x 256 -> 10 x 10 x 256
    out = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(base_encoder.output)
    out = Dropout(0.25)(out)
    out = Conv2D(256, (3, 3), activation="relu", padding="same")(out)
    out = Conv2D(256, (3, 3), activation="relu", padding="same")(out)
    out = BatchNormalization()(out)

    # 10 x 10 x 256 -> 20 x 20 x 128
    out = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(out)
    out = Dropout(0.25)(out)
    out = Conv2D(128, (3, 3), activation="relu", padding="same")(out)
    out = BatchNormalization()(out)

    # 20 x 20 x 128 -> 40 x 40 x 64
    out = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(out)
    out = Dropout(0.25)(out)
    out = Conv2D(64, (3, 3), activation="relu", padding="same")(out)
    out = BatchNormalization()(out)

    # 40 x 40 x 64 -> 40 x 40 x 1
    output = Conv2D(1, (1, 1), padding="same", activation='relu')(out)

    model = Model(inputs=base_encoder.input, outputs=output)
    model.summary()
    return model


get_model()
