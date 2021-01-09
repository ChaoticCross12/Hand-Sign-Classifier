# Importing libraries

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers import BatchNormalization
from keras import backend as K
import numpy as np
from keras.preprocessing import image
from keras.callbacks import History

import cv2

import matplotlib.pyplot as plt


# Model / data parameters

img_width, img_height = 300, 200
num_classes = 3
batch_size = 2
inputShape = (img_width, img_height, 3)

train_data_dir = "../dataset/RockPaperScissors"
validation_data_dir = "../dataset/RockPaperScissors"





# Data Augmentation

datagen = ImageDataGenerator(
    rescale = 1. / 255,
    validation_split = 0.25
)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    subset = "training",
    class_mode = "categorical"

)


val_datagen = ImageDataGenerator(
    rescale = 1. / 255
)

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    subset = "validation",
    class_mode = "categorical"
)



#THE MODEL

model = keras.Sequential()

# Set image shape: input_shape=(32, 32, 3)

model.add(Conv2D(32, kernel_size = (3, 3), input_shape=inputShape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.3))


model.add(Conv2D(32, kernel_size = (3, 3), input_shape=inputShape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size = (3, 3), input_shape=inputShape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = (3, 3), input_shape=inputShape))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.4))

model.add(Dense(3, activation='softmax'))

# Training

rms = keras.optimizers.RMSprop(learning_rate= 0.01, rho = 0.9)


# Loading the trained model




model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['categorical_accuracy'])

#history = History()
#
#
## fit   wda
##batch size = 2
#
#model.fit_generator(
#    train_generator,
#    steps_per_epoch = 821,
#    epochs = 20, callbacks = [history],
#    validation_data = validation_generator,
#    validation_steps = 656
#)
#
## Saving Model
#model.save_weights("modelNew.h5")
#print("Saved model to disk")
#
#
#plt.plot(history.history['categorical_accuracy'])
#plt.plot(history.history['val_categorical_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()


# Loading

model.load_weights("modelNew.h5")

img_pred = image.load_img('predTestR.png', target_size = (img_width, img_height))
img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis = 0)

rstl = model.predict(img_pred)
print(rstl)
