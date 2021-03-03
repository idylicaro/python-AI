import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

predictors = pd.read_csv('entradas_breast.csv')
targets = pd.read_csv('saidas_breast.csv')


def generate_model():
    model_classifier = keras.Sequential()
    model_classifier.add(layers.Dense(units=8, activation='relu',
                                      kernel_initializer='normal', input_dim=30))
    model_classifier.add(layers.Dropout(0.2))
    model_classifier.add(layers.Dense(units=8, activation='relu',
                                      kernel_initializer='normal'))
    model_classifier.add(layers.Dropout(0.2))
    model_classifier.add(layers.Dense(units=1, activation='sigmoid'))
    model_classifier.compile(optimizer='adam', loss='binary_crossentropy',
                             metrics=['binary_accuracy'])
    model_classifier.fit(predictors, targets, batch_size=10, epochs=1000)
    return model_classifier


def save_model(classifier, path='classifier_breast', weights_path='w_classifier_breast'):
    classifier_json = classifier.to_json()
    with open(path + '.json', 'w') as json_file:
        json_file.write(classifier_json)
    classifier.save_weights(weights_path + '.h5')


def load_model(path, weights_path):
    archive = open(path, 'r')
    network_config = archive.read()
    archive.close()

    classifier = model_from_json(network_config)
    classifier.load_weights(weights_path)
    return classifier


def execute(classifier, data):
    return classifier.predict(data)


model = generate_model()
save_model(model)

model = load_model('classifier_breast.json', 'w_classifier_breast.h5')
print('FINISH!')