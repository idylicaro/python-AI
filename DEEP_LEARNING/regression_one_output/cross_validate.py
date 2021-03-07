import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# pre-processing
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

base = pd.read_csv('autos.csv', encoding='ISO-8859-1')
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)
base = base.drop('name', axis=1)
base = base.drop('seller', axis=1)
base = base.drop('offerType', axis=1)

# inconsistent values
i1 = base.loc[base.price <= 10]
base.price.mean()
base = base[base.price > 10]
i2 = base.loc[base.price > 350000]
base = base.loc[base.price < 350000]

base.loc[pd.isnull(base['vehicleType'])]
base['vehicleType'].value_counts()  # limousine
base.loc[pd.isnull(base['gearbox'])]
base['gearbox'].value_counts()  # manuell
base.loc[pd.isnull(base['model'])]
base['model'].value_counts()  # golf
base.loc[pd.isnull(base['fuelType'])]
base['fuelType'].value_counts()  # benzin
base.loc[pd.isnull(base['notRepairedDamage'])]
base['notRepairedDamage'].value_counts()  # nein

values = {'vehicleType': 'limousine', 'gearbox': 'manuell',
          'model': 'golf', 'fuelType': 'benzin',
          'notRepairedDamage': 'nein'}

base = base.fillna(value=values)

predictors = base.iloc[:, 1:13].values
targets = base.iloc[:, 0].values

# transform categorical attribute
label_encoder_predictors = LabelEncoder()
predictors[:, 0] = label_encoder_predictors.fit_transform(predictors[:, 0])
predictors[:, 1] = label_encoder_predictors.fit_transform(predictors[:, 1])
predictors[:, 3] = label_encoder_predictors.fit_transform(predictors[:, 3])
predictors[:, 5] = label_encoder_predictors.fit_transform(predictors[:, 5])
predictors[:, 8] = label_encoder_predictors.fit_transform(predictors[:, 8])
predictors[:, 9] = label_encoder_predictors.fit_transform(predictors[:, 9])
predictors[:, 10] = label_encoder_predictors.fit_transform(predictors[:, 10])

# transform in dummy the categorical attribute
one_hot_encoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])],
                                    remainder='passthrough')
predictors = one_hot_encoder.fit_transform(predictors).toarray()


def create_network():
    regressor = Sequential()
    regressor.add(Dense(units=158, activation='relu', input_dim=316))
    regressor.add(Dense(units=158, activation='relu'))
    regressor.add(Dense(units=1, activation='linear'))
    regressor.compile(loss='mean_absolute_error', optimizer='adam',
                      metrics=['mean_absolute_error'])
    return regressor


regressor = KerasRegressor(build_fn=create_network,
                           epochs=100,
                           batch_size=300)
results = cross_val_score(estimator=regressor,
                          X=predictors, y=targets,
                          cv=10, scoring='neg_mean_absolute_error')
mean = results.mean()
desvio = results.std()

print(mean)
print(desvio)