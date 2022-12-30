import os, sys, shutil
from PIL import Image
import matplotlib.pyplot as plt
"""
TODO
1. preprocessing 정리
"""


# 원본 데이터셋을 압축 해제한 디렉터리 경로
original_dataset_dir = '../CNN_image/region/train'

# 소규모 데이터셋을 저장할 디렉터리
base_dir = '../CNN_image/train_small'
if os.path.exists(base_dir):  # 반복적인 실행을 위해 디렉토리를 삭제합니다.
    shutil.rmtree(base_dir)   # 이 코드는 책에 포함되어 있지 않습니다.
os.mkdir(base_dir)

# datagen = ImageDataGenerator(
#       rotation_range=40,
#       horizontal_flip=True,
#       fill_mode='nearest')

# 훈련, 검증, 테스트 분할을 위한 디렉터리
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
# test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

# 훈련용 rural 디렉터리
train_rural_dir = os.path.join(train_dir, 'rural')
os.mkdir(train_rural_dir)

# 훈련용 urban 사진 디렉터리
train_urban_dir = os.path.join(train_dir, 'urban')
os.mkdir(train_urban_dir)

# 검증용 rural 사진 디렉터리
validation_rural_dir = os.path.join(validation_dir, 'rural')
os.mkdir(validation_rural_dir)

# 검증용 urban 사진 디렉터리
validation_urban_dir = os.path.join(validation_dir, 'urban')
os.mkdir(validation_urban_dir)

import random

fnames_ru = ['rural_{}.png'.format(i) for i in range(1, 665)]
train_ru_fnames = random.sample(fnames_ru, 400)
fnames_ur = ['urban_{}.png'.format(i) for i in range(1, 452)]
train_ur_fnames = random.sample(fnames_ur, 400)
val_ru = random.sample(fnames_ru, 50)
val_ur = random.sample(fnames_ur, 50)
# len(train_ru_fnames)
# 처음 100개의 rural 이미지를 train_cats_dir에 복사합니다
for fname in train_ru_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_rural_dir, fname)
    shutil.copyfile(src, dst)

# 다음 32개 rural 이미지를 validation_cats_dir에 복사합니다
for fname in val_ru:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_rural_dir, fname)
    shutil.copyfile(src, dst)

# 처음 80개의 urban 이미지를 train_dogs_dir에 복사합니다
for fname in train_ur_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_urban_dir, fname)
    shutil.copyfile(src, dst)

# 다음 32개의 urban 이미지를 validation_dogs_dir에 복사합니다
for fname in val_ur:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_urban_dir, fname)
    shutil.copyfile(src, dst)




# # 다음 500개 강아지 이미지를 test_dogs_dir에 복사합니다
# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)
# from keras.preprocessing import image

# # 증식할 이미지 선택합니다
# img_path = fnames_ru[3]

# # 이미지를 읽고 크기를 변경합니다
# img = image.load_img(img_path, target_size=(500, 500))

# # (150, 150, 3) 크기의 넘파이 배열로 변환합니다
# x = image.img_to_array(img)

# # (1, 150, 150, 3) 크기로 변환합니다
# x = x.reshape((1,) + x.shape)

# # flow() 메서드는 랜덤하게 변환된 이미지의 배치를 생성합니다.
# # 무한 반복되기 때문에 어느 지점에서 중지해야 합니다!
# i = 0
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i % 4 == 0:
#         break


# plt.show()
print('훈련용 rural 이미지 전체 개수:', len(os.listdir(train_rural_dir)))
