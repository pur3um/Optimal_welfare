import os, shutil
import tensorflow as tf
import os, shutil
from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

# 원본 데이터셋을 압축 해제한 디렉터리 경로
original_dataset_dir = '../CNN_image/categorical/train'

# 소규모 데이터셋을 저장할 디렉터리
base_dir = '../CNN_image/train_small'
if os.path.exists(base_dir):  # 반복적인 실행을 위해 디렉토리를 삭제합니다.
    shutil.rmtree(base_dir)  # 이 코드는 책에 포함되어 있지 않습니다.
os.mkdir(base_dir)
# 훈련, 검증, 테스트 분할을 위한 디렉터리
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
# test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)
# 훈련용 디렉터리
train_field_dir = os.path.join(train_dir, 'field')
os.mkdir(train_field_dir)
train_island_dir = os.path.join(train_dir, 'island')
os.mkdir(train_island_dir)
train_mountain_dir = os.path.join(train_dir, 'mountain')
os.mkdir(train_mountain_dir)
train_urban_dir = os.path.join(train_dir, 'urban')
os.mkdir(train_urban_dir)

# 검증용 rural 사진 디렉터리
validation_field_dir = os.path.join(validation_dir, 'field')
os.mkdir(validation_field_dir)
validation_island_dir = os.path.join(validation_dir, 'island')
os.mkdir(validation_island_dir)
validation_mountain_dir = os.path.join(validation_dir, 'mountain')
os.mkdir(validation_mountain_dir)
validation_urban_dir = os.path.join(validation_dir, 'urban')
os.mkdir(validation_urban_dir)
import random

fnames_fi = ['field{}.png'.format(i) for i in range(1, 424)]
train_fi_fnames = random.sample(fnames_fi, 400)
fnames_is = ['island{}.png'.format(i) for i in range(1, 410)]
train_is_fnames = random.sample(fnames_is, 400)
fnames_mo = ['mountain{}.png'.format(i) for i in range(1, 418)]
train_mo_fnames = random.sample(fnames_mo, 400)
fnames_ur = ['urban{}.png'.format(i) for i in range(1, 452)]
train_ur_fnames = random.sample(fnames_ur, 400)
val_fi = random.sample(fnames_fi, 50)
val_is = random.sample(fnames_is, 50)
val_mo = random.sample(fnames_mo, 50)
val_ur = random.sample(fnames_ur, 50)
# len(train_ru_fnames)
# 각 카테고리별 이미지를 train에 복사합니다
for fname in train_fi_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_field_dir, fname)
    shutil.copyfile(src, dst)

for fname in val_fi:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_field_dir, fname)
    shutil.copyfile(src, dst)

for fname in train_is_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_island_dir, fname)
    shutil.copyfile(src, dst)

for fname in val_is:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_island_dir, fname)
    shutil.copyfile(src, dst)

for fname in train_mo_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_mountain_dir, fname)
    shutil.copyfile(src, dst)

for fname in val_mo:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_mountain_dir, fname)
    shutil.copyfile(src, dst)

for fname in train_ur_fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_urban_dir, fname)
    shutil.copyfile(src, dst)

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
print('훈련용 rural 이미지 전체 개수:', len(os.listdir(train_mountain_dir)))
훈련용
rural
이미지
전체
개수: 400
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
# from keras.models import Sequential
# from keras.layers import Dropout, Activation, Dense
# from keras.layers import Flatten, Convolution2D, MaxPooling2D
# from keras.models import load_model

model = models.Sequential()

model.add(layers.Convolution2D(16, (3, 3), activation='relu', input_shape=(707, 1000, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Convolution2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Convolution2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Convolution2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))
model.summary()

# model.compile(optimizer='adam',
#              loss = 'categorical_crossentropy',
#              metrics=['acc'])
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True)

# 모든 이미지를 1/255로 스케일을 조정합니다
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # 타깃 디렉터리
        train_dir,
        # 모든 이미지를 150 × 150 크기로 바꿉니다
        target_size=(707, 1000),
        batch_size=10,
        # binary_crossentropy 손실을 사용하기 때문에 이진 레이블이 필요합니다
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(707, 1000),
        batch_size=10,
        class_mode='categorical')

for data_batch, labels_batch in train_generator:
    print('배치 데이터 크기:', data_batch.shape)
    print('배치 레이블 크기:', labels_batch.shape)
    break

# 160, 20
history = model.fit_generator(
    train_generator,
    steps_per_epoch=160,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=20)

model.save('region3.h5')
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

example_dir='../CNN_image/train_test/test/'
example_datagenerator = ImageDataGenerator(rescale=1./255)
example_set = example_datagenerator.flow_from_directory(
    example_dir,
    batch_size =10,
    target_size=(707,1000),
    class_mode='categorical'
    )

example_loss, example_acc = model.evaluate_generator(example_set, steps=1)
print('test acc:', example_acc)

output = model.predict_generator(test_set, steps=20)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
print(test_set.class_indices)
print(output)
print(test_set.filenames)

test_dir = '../CNN_image/test'
test_datagenerator = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagenerator.flow_from_directory(
    test_dir,
    batch_size=10,
    target_size=(707, 1000),
    class_mode='categorical'
)
len(test_set)
test_loss, test_acc = model.evaluate_generator(test_set, steps=20)
print('test acc:', test_acc)
output = model.predict_generator(test_set, steps=20)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
print(test_set.class_indices)
print(output)
print(test_set.filenames)
# test_loss, test_acc = model.evaluate_generator(test_set, steps=2)
# print('test acc:', test_acc)
# from keras.models import Sequential
# from keras.layers import Dropout, Activation, Dense
# from keras.layers import Flatten, Convolution2D, MaxPooling2D
# from keras.models import load_model

# model1 = models.Sequential()

# model1.add(layers.Convolution2D(16,(3,3), activation='relu', input_shape=(707,1000,3)))
# model1.add(layers.MaxPooling2D((2,2)))
# model1.add(layers.Dropout(0.25))
# model1.add(layers.Convolution2D(32,(3,3), activation='relu'))
# model1.add(layers.MaxPooling2D((2,2)))
# model1.add(layers.Dropout(0.25))
# model1.add(layers.Convolution2D(32,(3,3), activation='relu'))
# model1.add(layers.MaxPooling2D((2,2)))
# model1.add(layers.Dropout(0.25))
# model1.add(layers.Convolution2D(32,(3,3), activation='relu'))
# model1.add(layers.MaxPooling2D((2,2)))
# model1.add(layers.Dropout(0.25))
# model1.add(layers.Convolution2D(32,(3,3), activation='relu'))
# model1.add(layers.MaxPooling2D((2,2)))
# model1.add(layers.Dropout(0.25))
# model1.add(layers.Convolution2D(64,(3,3), activation='relu'))
# model1.add(layers.MaxPooling2D((2,2)))
# model1.add(layers.Dropout(0.25))
# model1.add(layers.Flatten())
# model1.add(layers.Dense(512,activation='relu'))
# model1.add(layers.Dropout(0.5))
# model1.add(layers.Dense(4,activation='softmax'))
# model1.summary()
# model1.compile(optimizer='adam',
#              loss = 'categorical_crossentropy',
#              metrics=['acc'])
# pred = model1.predict(test_set)
# pred
# test_loss, test_acc = model1.evaluate_generator(test_set, steps=1)
# print('test acc:', test_acc)
# output = model1.predict_generator(test_set, steps=5)
# print(test_set.class_indices)
# print(output)
# print(test_set.filenames)