import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

base = pd.read_csv('iris.csv')
predictors = base.iloc[:, 0:4].values
targets = base.iloc[:, 4].values    # iris setosa 1 0 0, iris versicolor 0 1 0, iris virginica 0 0 1


# organize targets for the mult classify
label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(targets)
targets_dummy = to_categorical(targets)


predictors_train, predictors_test, targets_train, targets_test = train_test_split(predictors, targets_dummy, test_size=0.25)

classifier = keras.Sequential()

classifier.add(Dense(units=4, activation='relu', input_dim=4))
classifier.add(Dense(units=4, activation='relu'))
classifier.add(Dense(units=3, activation='softmax'))  # softmax to classify more than 2 classes

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
classifier.fit(predictors_train, targets_train, batch_size=10, epochs=1000)

result = classifier.evaluate(predictors_test, targets_test)