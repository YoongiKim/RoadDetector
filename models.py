import keras
from keras import layers as L
from keras import Model, Sequential
import tensorflow as tf
from keras.layers import TimeDistributed as Dist

def build_model():
    ch = 32
    kernel = (3, 3)
    stride1 = (1, 1)
    stride2 = (2, 2)

    i = L.Input((288, 512, 3))
    e1 = L.Conv2D(ch * 1, kernel_size=kernel, strides=(1, 1), padding='same')(i)
    e1 = L.BatchNormalization()(e1)
    e1 = L.Activation('relu')(e1)
    e2 = L.MaxPooling2D()(e1)

    e2 = L.Conv2D(ch * 1, kernel_size=kernel, strides=stride1, padding='same')(e2)
    e2 = L.BatchNormalization()(e2)
    e2 = L.Activation('relu')(e2)
    e3 = L.MaxPooling2D()(e2)

    e3 = L.Conv2D(ch * 2, kernel_size=kernel, strides=stride1, padding='same')(e3)
    e3 = L.BatchNormalization()(e3)
    e3 = L.Activation('relu')(e3)
    e4 = L.MaxPooling2D()(e3)

    e4 = L.Conv2D(ch * 4, kernel_size=kernel, strides=stride1, padding='same')(e4)
    e4 = L.BatchNormalization()(e4)
    e4 = L.Activation('relu')(e4)
    e5 = L.MaxPooling2D()(e4)

    e5 = L.Conv2D(ch * 8, kernel_size=kernel, strides=stride1, padding='same')(e5)
    e5 = L.BatchNormalization()(e5)
    e5 = L.Activation('relu')(e5)
    e6 = L.MaxPooling2D()(e5)

    e6 = L.Conv2D(ch * 16, kernel_size=kernel, strides=stride1, padding='same')(e6)
    e6 = L.BatchNormalization()(e6)
    e6 = L.Activation('relu')(e6)

    d6 = L.Conv2DTranspose(ch * 16, kernel_size=kernel, strides=stride2, padding='same')(e6)
    d6 = L.BatchNormalization()(d6)
    d6 = L.Activation('relu')(d6)
    d6 = L.Concatenate()([d6, e5])

    d5 = L.Conv2DTranspose(ch * 8, kernel_size=kernel, strides=stride2, padding='same')(d6)
    d5 = L.BatchNormalization()(d5)
    d5 = L.Activation('relu')(d5)
    d5 = L.Concatenate()([d5, e4])

    d4 = L.Conv2DTranspose(ch * 4, kernel_size=kernel, strides=stride2, padding='same')(d5)
    d4 = L.BatchNormalization()(d4)
    d4 = L.Activation('relu')(d4)
    d4 = L.Concatenate()([d4, e3])

    d3 = L.Conv2DTranspose(ch * 2, kernel_size=kernel, strides=stride2, padding='same')(d4)
    d3 = L.BatchNormalization()(d3)
    d3 = L.Activation('relu')(d3)
    d3 = L.Concatenate()([d3, e2])

    d2 = L.Conv2DTranspose(ch * 1, kernel_size=kernel, strides=stride2, padding='same')(d3)
    d2 = L.BatchNormalization()(d2)
    d2 = L.Activation('relu')(d2)
    d2 = L.Concatenate()([d2, e1])

    d1 = L.Conv2D(3, kernel_size=(1, 1), strides=(1, 1))(d2)
    d1 = L.BatchNormalization()(d1)
    d1 = L.Activation('softmax')(d1)

    model = Model(inputs=[i], outputs=[d1])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', 'mae'])
    model.summary()

    return model

# def build_model2(height=720, width=1280, kernel=80, stride=80):
#     i = L.Input(batch_shape=(None, height, width, 3))
#     x = L.Lambda(lambda x: tf.extract_image_patches(
#         x, ksizes=[1, kernel, kernel, 1], strides=[1, stride, stride, 1], rates=[1, 1, 1, 1], padding='VALID'))(i)
#
#     out_width = int((width - kernel)/stride + 1)
#     out_height = int((height - kernel)/stride + 1)
#     print(out_height, out_width)
#
#     x = L.Reshape([out_height, out_width, kernel, kernel, 3])(x)
#     x = L.Reshape([out_height*out_width, kernel, kernel, 3])(x)  # [144, 80, 80, 3]
#
#     x = Dist(L.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))(x)
#     x = Dist(L.BatchNormalization())(x)
#     x = Dist(L.Activation('relu'))(x)
#     x = Dist(L.MaxPool2D())(x)
#
#     x = Dist(L.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))(x)
#     x = Dist(L.BatchNormalization())(x)
#     x = Dist(L.Activation('relu'))(x)
#     x = Dist(L.MaxPool2D())(x)
#
#     x = Dist(L.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))(x)
#     x = Dist(L.BatchNormalization())(x)
#     x = Dist(L.Activation('relu'))(x)
#     x = Dist(L.MaxPool2D())(x)
#
#     x = Dist(L.Conv2D(512, (3, 3), strides=(2, 2), padding='same'))(x)
#     x = Dist(L.BatchNormalization())(x)
#     x = Dist(L.Activation('relu'))(x)
#
#     x = Dist(L.Flatten())(x)
#     x = L.Bidirectional(L.CuDNNLSTM(64, return_sequences=True))(x)
#     x = L.Bidirectional(L.CuDNNLSTM(64, return_sequences=True))(x)
#     x = Dist(L.Dense(3, activation='softmax'))(x)
#     x = L.Reshape([out_height, out_width, 3])(x)
#
#     model = Model(inputs=[i], outputs=[x])
#
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', 'mae'])
#     model.summary()
#
#     return model
