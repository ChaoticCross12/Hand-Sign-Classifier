# Importing libraries

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image


# Model / data parameters

img_width, img_height = 300, 200
num_classes = 3
batch_size = 20

train_data_dir = "./dataset"
validation_data_dir = "./dataset"

# Setting input shape
#
#if K.input_data_format() == 'channels_first':
#    input_shape = (3, img_width, img_height)
#
#else:
#    input_shape = (img_width, img_height, 3)
#














# Data Augmentation

datagen = ImageDataGenerator(
    rescale = 1. / 255
    validation_split = 0.3
)

train_generator = datagen.flow_form_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    subset = "training",
    class_mode = "categorical"

)


val_datagen = ImageDataGenerator(
    rescale = 1. / 255
)

validaation_generator = datagen.flow_form_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    subset = "validaiton",
    class_mode = "categorical"    
)






#THE MODEL

model = keras.Sequential()

model.add(Conv2D(32, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu')
model.add(Dropout(0.5))
model.add(Dense(1))

model.add(Activation = 'sigmoid')

model.summary()


# Training

rms = keras.optimizer

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])



# Evaluating 

score = model.evaluate(x_test, y_test, verbose = 0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])