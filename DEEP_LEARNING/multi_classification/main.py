import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

base = pd.read_csv('iris.csv')
predictors = base.iloc[:, 0:4].values
targets = base.iloc[:, 4].values
label_encoder = LabelEncoder()
targets = label_encoder.fit_transform(targets)
targets_dummy = to_categorical(targets)


def create_network(optimizer, loss, kernel_initializer, activation, neurons):
    network = Sequential()
    network.add(Dense(units=4, activation=activation, kernel_initializer=kernel_initializer, input_dim=4))
    network.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    network.add(Dense(units=3, activation='softmax'))
    network.compile(optimizer=optimizer, loss=loss,
                    metrics=['categorical_accuracy'])
    return network


classifier = KerasClassifier(build_fn=create_network)
parameters = {
    'batch_size': [10, 20],
    'epochs': [2000, 5000],
    'optimizer': ['adam', 'sgd'],
    'loss': ['categorical_crossentropy'],
    'kernel_initializer': ['random_uniform', 'normal'],
    'activation': ['relu', 'tanh'],
    'neurons': [2, 3, 4]}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)

grid_search = grid_search.fit(predictors, targets)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_params)
