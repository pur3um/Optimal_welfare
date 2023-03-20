import os
import shutil

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

from keras import datasets, layers, models
from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D
from keras.models import load_model


def segmentation_classification():
    model = models.Sequential()

    model.add(Convolution2D(16, (3, 3), activation='relu', input_shape=(707, 1000, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.summary()

    # model.compile(optimizer='adam',
    #              loss = 'categorical_crossentropy',
    #              metrics=['acc'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    model.save('region3.h5')

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

    return model

# Test Check
model = segmentation_classification()

test_dir = '../CNN_image/test'
test_datagenerator = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagenerator.flow_from_directory(test_dir,
                                                  batch_size=10,
                                                  target_size=(707, 1000),
                                                  class_mode='categorical'
                                                  )

test_loss, test_acc = model.evaluate_generator(test_set, steps=1)
print('test acc:', test_acc)

output = model.predict_generator(test_set, steps=20)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

print(test_set.class_indices)
print(output)
print(test_set.filenames)

print(len(test_set))
test_loss, test_acc = model.evaluate_generator(test_set, steps=20)
print('test acc:', test_acc)
output = model.predict_generator(test_set, steps=20)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

print(test_set.class_indices)
print(output)
print(test_set.filenames)