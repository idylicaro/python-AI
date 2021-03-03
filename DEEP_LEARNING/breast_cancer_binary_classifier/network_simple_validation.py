import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

predictors = pd.read_csv('entradas_breast.csv')
targets = pd.read_csv('saidas_breast.csv')

# split train data
predictors_train, predictors_test, targets_train, targets_test = train_test_split(predictors, targets, test_size=0.25)

model_classifier = keras.Sequential()

# create first hidden layer
# units = (count_inputs + count_output) / 2
model_classifier.add(layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform', input_dim=30))
# create second hidden layer
model_classifier.add(layers.Dense(units=16, activation='relu', kernel_initializer='random_uniform'))

# create output layer
model_classifier.add(layers.Dense(units=1, activation='sigmoid'))


# model configuration
# optimizer is a gradiant method
optimizer = keras.optimizers.Adam(lr=0.001, decay=0.0001, clipvalue=0.5)
model_classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

# model train
model_classifier.fit(predictors_train, targets_train, batch_size=10, epochs=100)

# weights of model
weight_01 = model_classifier.layers[0].get_weights()
weight_12 = model_classifier.layers[1].get_weights()
weight_2 = model_classifier.layers[2].get_weights()

# manual test
predictions = model_classifier.predict(predictors_test)
predictions = ( predictions > 0.5)
accuracy = accuracy_score(targets_test, predictions)
matrix = confusion_matrix(targets_test, predictions)

# result = model_classifier.evaluate(predictors_test, targets_test)
