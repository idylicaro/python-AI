import pandas as pd
from sklearn.model_selection import cross_val_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

predictors = pd.read_csv('entradas_breast.csv')
targets = pd.read_csv('saidas_breast.csv')


def create_network(optimizer, loss, kernel_initializer, activation, neurons):
    network = keras.Sequential()

    # create first hidden layer
    # units = (count_inputs + count_output) / 2
    network.add(layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=30))
    # add dropout
    network.add(layers.Dropout(rate=0.2))
    # create second hidden layer
    network.add(layers.Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    # add dropout
    network.add(layers.Dropout(rate=0.2))
    # create output layer
    network.add(layers.Dense(units=1, activation='sigmoid'))

    # model configuration
    # optimizer is a gradiant method
    network.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return network


# Tunning method
model_classifier = KerasClassifier(build_fn=create_network)
parameters = {
    'batch_size': [10, 30],
    'epochs': [100, 200],
    'optimizer': ['adam', 'sgd'],
    'loss': ['binary_crossentropy', 'hinge'],
    'kernel_initializer': ['random_uniform', 'normal'],
    'activation': ['relu', 'tanh'],
    'neurons': [16, 22]
}
grid_search = GridSearchCV(estimator=model_classifier, param_grid=parameters, scoring='accuracy', cv=5)

grid_search = grid_search.fit(predictors, targets)
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_params)

#results = cross_val_score(estimator=model_classifier, X=predictors, y=targets, cv=10, scoring='accuracy')

#mean = results.mean()
#standard_deviation = results.std()
#print(f'accuracy:({mean * 100.0}%)\nstandard deviation:({standard_deviation})')
