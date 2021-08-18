import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

seed = 5
np.random.seed(seed)

(X, y), (x_test, y_test) = mnist.load_data()

# if you can see something image use -> plt.imshow(x_training[n])
# plt.imshow(x_training[0])

# Change the matrix to a channel, so just black and white
predictors = X.reshape(X.shape[0], 28, 28, 1)

# for the normalization 0 - 1
predictors = predictors.astype('float32')

# min max normalization
predictors /= 255

# dummy
targets = utils.to_categorical(y, 10)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = []

for index_training, index_test in kfold.split(predictors, np.zeros(shape=(targets.shape[0], 1))):
    classifier = Sequential()

    # -- neural convolution --

    # Conv2d(num_of_kernels, size_of_feature_detector, ...)
    # is recommended use 64 kernels
    # Convolution Operator
    classifier.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # normalization in feature maps layer
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolution Operator
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())

    # -- neural convolution --

    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=10, activation='softmax'))
    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    classifier.fit(predictors[index_training], targets[index_training], batch_size=128, epochs=5)
    precision = classifier.evaluate(predictors[index_test], targets[index_test])
    results.append(precision[1])

print('Mean: ', sum(results) / len(results))
