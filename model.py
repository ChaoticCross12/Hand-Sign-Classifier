<<<<<<< HEAD
# Importing libraries

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
import numpy as np
from keras.preprocessing import image
from keras.callbacks import History


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

rms = keras.optimizers.RMSprop(learning_rate= 0.5, rho = 0.9)

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['categorical_accuracy'])



history = History()

# fit   wda
#batch size = 2

model.fit_generator(
    train_generator,
    steps_per_epoch = 821,
    epochs = 25, callbacks = [history],
    validation_data = validation_generator,
    validation_steps = 656
)

model.summary()

=======
# Importing libraries

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.preprocessing import image
from keras.callbacks import History


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
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, kernel_size = (3, 3), input_shape=inputShape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, kernel_size = (3, 3), input_shape=inputShape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, kernel_size = (3, 3), input_shape=inputShape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.add(Activation('sigmoid'))


# Training

rms = keras.optimizers.RMSprop(learning_rate= 0.5, rho = 0.9)

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['categorical_accuracy'])



history = History()

# fit   wda
#batch size = 2

model.fit_generator(
    train_generator,
    steps_per_epoch = 821,
    epochs = 30, callbacks = [history],
    validation_data = validation_generator,
    validation_steps = 656
)



model.summary()


# Evaluating 

#score = model.evaluate(x_test, y_test, verbose = 0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])
>>>>>>> 70efb15aaf5de74bb7af2abde7def703fe3dd46c
