import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

predictors = pd.read_csv('entradas_breast.csv')
targets = pd.read_csv('saidas_breast.csv')


def create_network():
    network = keras.Sequential()

    # create first hidden layer
    # units = (count_inputs + count_output) / 2
    network.add(layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
    # add dropout
    network.add(layers.Dropout(rate=0.2))
    # create second hidden layer
    network.add(layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    # add dropout
    network.add(layers.Dropout(rate=0.2))
    # create output layer
    network.add(layers.Dense(units=1, activation='sigmoid'))

    # model configuration
    # optimizer is a gradiant method
    optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
    network.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    return network


model_classifier = KerasClassifier(build_fn=create_network, epochs=100, batch_size=10)

results = cross_val_score(estimator=model_classifier, X=predictors, y=targets, cv=10, scoring='accuracy')

mean = results.mean()
standard_deviation = results.std()
print(f'accuracy:({mean*100.0}%)\nstandard deviation:({standard_deviation})')