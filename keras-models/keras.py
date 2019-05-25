#-*- coding: UTF-8 -*-
# ref: https://blog.csdn.net/liuuze5/article/details/79529880

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout

model = Sequential()
# the first layer need to specify input_shape parameter.
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))

model.add(MaxPooling2D(pool_size(2,2)))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Flatten())

from keras.optimizers import SGD

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size = 32, epochs = 10, validation_data(x_val, y_val))

score = model.evaluate(x_test, y_test, batch_size = 32)


model.save_weights("my_model.h5")
model.load_weights('my_model_weights.h5')


## ----

from keras.models import Model

from keras.layers import Input

digit_input = Input(shape=(1,28,28))

x=Conv2D(64,(3,3))(digit_input)
x=Conv2D(64,(3,3))(x)
x=MaxPooling2D((2,2))(x)
out=Flatten()(x)

model = Model(digit_input, out)



