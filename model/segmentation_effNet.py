from keras.preprocessing.image import ImageDataGenerator
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam
from math import ceil
from time import time
import keras.backend as K
import tensorflow as tf
from tensorflow_addons.layers import GroupNormalization
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
https://keras.io/api/applications/efficientnet/
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

        인수
        include_top : 네트워크 상단에 완전 연결 계층을 포함할지 여부. 기본값은 True입니다.
        weights : None(무작위 초기화), 'imagenet'(ImageNet에 대한 사전 교육) 또는 로드할 가중치 파일의 경로 중 하나입니다.
        기본값은 '이미지넷'입니다.
        input_tensorlayers.Input() : 모델의 이미지 입력으로 사용할 선택적 Keras 텐서(예: 의 출력 ).
        input_shapeinclude_top : 선택적 모양 튜플, False 인 경우에만 지정됩니다 . 정확히 3개의 입력 채널이 있어야 합니다.
        pooling : 기능 추출을 위한 선택적 풀링 모드 include_top입니다 False. 기본값은 없음입니다.
         - None모델의 출력이 마지막 컨볼루션 레이어의 4D 텐서 출력임을 의미합니다.
          - avg전역 평균 풀링이 마지막 컨볼루션 레이어의 출력에 적용되므로 모델의 출력은 2D 텐서가 됩니다.
           - max전역 최대 풀링이 적용됨을 의미합니다.
        classes : 이미지를 분류할 선택적 클래스 수이며 include_top, True인 경우와 weights인수가 지정되지 않은 경우에만 지정됩니다.
         기본값은 1000(ImageNet 클래스 수)입니다.
        classifier_activation : A str또는 호출 가능. "최상위" 레이어에서 사용할 활성화 함수입니다. 가 아니면 무시됩니다 include_top=True.
         classifier_activation=None"최상위" 레이어의 로짓을 반환하도록 설정 합니다. 기본값은 'softmax'입니다.
          사전 훈련된 가중치를 로드할 때 는 또는 classifier_activation이어야 합니다 .None"softmax"
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