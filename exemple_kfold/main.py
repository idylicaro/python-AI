from collections import Counter
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

data_frame = pd.read_csv('situacao_do_cliente.csv')
X_dataframe = data_frame[['recencia', 'frequencia', 'semanas_de_inscricao']]
Y_dataframe = data_frame['situacao']

X_dummies_dataframe = pd.get_dummies(X_dataframe)
Y_dummies_dataframe = Y_dataframe

X = X_dummies_dataframe.values
Y = Y_dummies_dataframe.values

training_percentage = 0.8

training_size = int(training_percentage * len(Y))

data_training = X[:training_size]
data_training_markings = Y[:training_size]

data_of_validation = X[training_size:]
data_of_validation_markings = Y[training_size:]


def fit_and_predict(name, model, x_training, y_training):
    k = 10
    scores = cross_val_score(model, x_training, y_training, cv=k)
    hit_rate = 100.0 * np.mean(scores)

    msg = "Hit rate of {0}: {1}".format(name, hit_rate)
    print(msg)
    return hit_rate


def real_test(model, _data_of_validation, _data_of_validation_markings):
    result = model.predict(_data_of_validation)

    hits = result == _data_of_validation_markings

    total_of_hits = sum(hits)
    total_of_elements = len(_data_of_validation_markings)

    hits_rate = 100.0 * total_of_hits / total_of_elements

    msg = "Winner hit rate among real-world algorithms : {0}".format(hits_rate)
    print(msg)


results = {}

modelOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0, max_iter=10000))
resultOneVsRest = fit_and_predict("OneVsRest", modelOneVsRest, data_training, data_training_markings)
results[resultOneVsRest] = modelOneVsRest

modelOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=10000))
resultOneVsOne = fit_and_predict("OneVsOne", modelOneVsOne, data_training, data_training_markings)

results[resultOneVsOne] = modelOneVsOne

modelMultinomial = MultinomialNB()
resultMultinomial = fit_and_predict("MultinomialNB", modelMultinomial, data_training, data_training_markings)
results[resultMultinomial] = modelMultinomial

modelAdaBoost = AdaBoostClassifier()
resultAdaBoost = fit_and_predict("AdaBoostClassifier", modelAdaBoost, data_training, data_training_markings)
results[resultAdaBoost] = modelAdaBoost

better = max(results)
winner = results[better]

print("Winner: ")
print(winner)

winner.fit(data_training, data_training_markings)

real_test(winner, data_of_validation, data_of_validation_markings)
base_hits = max(Counter(data_of_validation_markings).values())

base_hits_rate = 100.0 * base_hits / len(data_of_validation_markings)
print("Base Hits Rate: %f" % base_hits_rate)

total_of_elements = len(data_of_validation)
print("Total of tests: %d" % total_of_elements)
