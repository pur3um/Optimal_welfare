import os, shutil
import tensorflow as tf
import os, shutil
from keras import optimizers
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

#
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