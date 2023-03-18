# rural urban Classification
import keras
import os, shutil
import tensorflow as tf
from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator

# 모든 이미지를 1/255로 스케일을 조정합니다
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        # 타깃 디렉터리
        train_dir,
        # 모든 이미지를 150 × 150 크기로 바꿉니다
        target_size=(707, 1000),
        batch_size=10,
        # binary_crossentropy 손실을 사용하기 때문에 이진 레이블이 필요합니다
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(707, 1000),
        batch_size=10,
        class_mode='binary')

for data_batch, labels_batch in train_generator:
    print('배치 데이터 크기:', data_batch.shape)
    print('배치 레이블 크기:', labels_batch.shape)
    break

from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model

model = models.Sequential()

model.add(layers.Convolution2D(16,(3,3), activation='relu', input_shape=(707,1000,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Convolution2D(32,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Convolution2D(32,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Convolution2D(32,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Convolution2D(32,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Convolution2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics=['acc'])
history = model.fit_generator(
      train_generator,
      steps_per_epoch=50,
      epochs=20,
      validation_data=validation_generator,
      validation_steps=25)

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

from keras import backend as K
# test_loss, test_acc = model.evaluate_generator(test_set, steps=50)
# print('test acc:', test_acc)

"""
datagen = ImageDataGenerator(
      rotation_range=40,
      horizontal_flip=True,
      fill_mode='nearest')
# 이미지 전처리 유틸리티 모듈
from keras.preprocessing import image

fnames = sorted([os.path.join(train_rural_dir, fname) for fname in os.listdir(train_rural_dir)])

# 증식할 이미지 선택합니다
img_path = fnames[3]

# 이미지를 읽고 크기를 변경합니다
img = image.load_img(img_path, target_size=(512, 512))

# (150, 150, 3) 크기의 넘파이 배열로 변환합니다
x = image.img_to_array(img)

# (1, 150, 150, 3) 크기로 변환합니다
x = x.reshape((1,) + x.shape)

# flow() 메서드는 랜덤하게 변환된 이미지의 배치를 생성합니다.
# 무한 반복되기 때문에 어느 지점에서 중지해야 합니다!
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()
"""

test_dir = '../CNN_image/test'
test_datagenerator = ImageDataGenerator(rescale=1./255)
test_set = test_datagenerator.flow_from_directory(
    test_dir,
    batch_size =10,
    target_size=(707,1000),
    class_mode='binary'
    )

pred = model.predict(test_set)

print(pred)

test_loss, test_acc = model.evaluate_generator(test_set, steps=10)
print('test acc:', test_acc)

model1 = models.Sequential()

model1.add(layers.Convolution2D(16,(3,3), activation='relu', input_shape=(707,1000,3)))
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Dropout(0.25))
model1.add(layers.Convolution2D(32,(3,3), activation='relu'))
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Dropout(0.25))
model1.add(layers.Convolution2D(32,(3,3), activation='relu'))
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Dropout(0.25))
model1.add(layers.Convolution2D(32,(3,3), activation='relu'))
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Dropout(0.25))
model1.add(layers.Convolution2D(64,(3,3), activation='relu'))
model1.add(layers.MaxPooling2D((2,2)))
model1.add(layers.Dropout(0.25))
model1.add(layers.Flatten())
model1.add(layers.Dense(512,activation='relu'))
model1.add(layers.Dropout(0.5))
model1.add(layers.Dense(1,activation='sigmoid'))
model1.summary()

model1.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics=['acc'])
pred = model1.predict(test_set)
# history = model1.fit_generator(
#       test_set,
#       steps_per_epoch=50,
#       epochs=11)
test_loss, test_acc = model1.evaluate_generator(test_set, steps=1)
print('test acc:', test_acc)

output = model1.predict_generator(test_set, steps=5)
print(test_set.class_indices)
print(output)
# print(len(test_set.filenames))
print(test_set.filenames)

