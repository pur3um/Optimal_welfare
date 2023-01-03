import os, sys, shutil
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
from keras.preprocessing import image
import matplotlib.pyplot as plt
"""
TODO
1. preprocessing 정리
"""


# 원본 데이터셋을 압축 해제한 디렉터리 경로
# gfz PC, home PC, laptop : directory path 확인
original_dataset_dir = '../input/region/train'

class DataPreprocessing:
    def __init(self):
        self.image_path = "../input/train_small"  # 정확한 path 측정후 수정 필요
        self.file_name = None
        self.file_type = None

    def state_check(self):
        return np.shape(self.image_path)

    def data_generator(self, x_data, y_data, batch_size):
        """
        원본 이미지는 0-255의 RGB 계수로 구성
        이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높음 (통상적인 learning rate를 사용할 경우).
        그래서 이를 1/255로 스케일링하여 0-1 범위로 변환. 이는 다른 전처리 과정에 앞서 가장 먼저 적용
        """
        datagen = ImageDataGenerator(rescale=1./255,  # 값을 0과 1사이로 변경
                                     rotation_range=20,  # 무작위 회전 각도 20도 이내
                                     horizontal_flip=True,  # 무작위로 가로 뒤짚음
                                     )
        generator = datagen.flow(  # 변형된 이미지를 배치 단위로 불러올 수 있는 generator 생성
            x=x_data, y=y_data,
            batch_size=batch_size,
            shuffle=True
        )

        return generator

    def reshape_image(self, file_path):
        img = load_img(file_path)

        img_array = img_to_array(img)  # (3, 150, 150) size Numpy array
        # convert to (1, 3, 150, 150) size
        img_array = img_array.reshape((1,) + img_array.shape)
        return img_array


