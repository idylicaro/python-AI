import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('iris.csv')
predictors = base.iloc[:, 0:4].values
targets = base.iloc[:, 4].values
labelencoder = LabelEncoder()
targets = labelencoder.fit_transform(targets)
targets_dummy = to_categorical(targets)


def create_network():
    classifier = Sequential()
    classifier.add(Dense(units=4, activation='relu', input_dim=4))
    classifier.add(Dense(units=4, activation='relu'))
    classifier.add(Dense(units=3, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                       metrics=['categorical_accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=create_network,
                             epochs=1000,
                             batch_size=10)
results = cross_val_score(estimator=classifier,
                          X=predictors, y=targets,
                          cv=10, scoring='accuracy')
mean = results.mean()
deviation = results.std()

print(mean*100.0)