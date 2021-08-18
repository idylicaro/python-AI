import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils


(x_training, y_training), (x_test, y_test) = mnist.load_data()

# if you can see something image use -> plt.imshow(x_training[n])
plt.imshow(x_training[0])

# Change the matrix to a channel, so just black and white
predictors_training = x_training.reshape(x_training.shape[0], 28, 28, 1)
predictors_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# for the normalization 0 - 1
predictors_training = predictors_training.astype('float32')
predictors_test = predictors_test.astype('float32')

# min max normalization
predictors_training /= 255
predictors_test /= 255

# dummy
targets_training = utils.to_categorical(y_training, 10)
targets_test = utils.to_categorical(y_test, 10)

classifier = Sequential()

# -- neural convolution --

# Conv2d(num_of_kernels, size_of_feature_detector, ...)
# is recommended use 64 kernels
# Convolution Operator
classifier.add(Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu' ))
# normalization in feature maps layer
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Convolution Operator
classifier.add(Conv2D(32, (3,3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())

# -- neural convolution --

classifier.add(Dense(units= 128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units= 10, activation='softmax'))
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier.fit(predictors_training, targets_training, batch_size=128, epochs=5, validation_data=(predictors_test, targets_test))
result = classifier.evaluate(predictors_test, targets_test)