from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam
from math import ceil
from time import time
import keras.backend.tensorflow_backend as K
import tensorflow as tf
import efficientnet.keras as efn
import numpy as np
import argparse
import random
import keras
import sys
import cv2
import csv
import os

"""
왜 EfficientNet을 사용했는지, : effnet의 장점
데이터의 포맷과 관련해서 버전에 왜 이게 더 맞는지
- efficientNet은 inputsize = 600
-> fine tuning으로 진행
또한 efficientNet은 scale이 되어있기 때문에 scale 사용하면 안됨,
"""

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# K.set_session(tf.Session(config=config))


class EFNet():
    def __init__(self, output_size, batch_size, ef_nm, img_shape, sample_size):
        self.output_size = output_size
        self.batch_size = batch_size
        self.ef_nm = ef_nm
        self.weight_path = "weight/" + self.ef_nm + ".h5"
        self.log_path = "log/" + self.ef_nm + ".csv"
        self.img_shape = img_shape
        self.sample_size = sample_size
        # self.graph = tf.get_default_graph()

        effnet = self.getEFNet()
        self.model = self.buildModel(effnet)

    def getEFNet(self):
        """
        tf.keras.applications.EfficientNetB7(
                            include_top=True,
                            weights="imagenet",
                            input_tensor=None,
                            input_shape=None,
                            pooling=None,
                            classes=1000,
                            classifier_activation="softmax",
                            **kwargs
                        )
        """
        effnet = efn.EfficientNetB4(weights=None,
                                    include_top=False,
                                    input_shape=self.img_shape)
        #         effnet.load_weights('weight/efficientnet-b4_imagenet_1000_notop.h5')

        for i, layer in enumerate(effnet.layers):
            if "batch_normalization" in layer.name:
                effnet.layers[i] = GroupNormalization(groups=self.batch_size, axis=-1, epsilon=0.00001)

        return effnet

    def buildModel(self, effnet):
        model = Sequential()

        model.add(effnet)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(self.output_size, activation='sigmoid'))

        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=Adam(lr=0.00005),
                      metrics=['acc'])

        return model


if __name__ == "__main__":
    efnet = EFNet(255, 8, 'name', (256, 256, 3), 1)