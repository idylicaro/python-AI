import nltk
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

data_frame = pd.read_csv('emails.csv', encoding='utf-8')

data_texts = data_frame['email']
sentences = data_texts.str.lower()

nltk.download("punkt")
broked_texts = [nltk.tokenize.word_tokenize(frase) for frase in sentences]

dictionary = set()

# Removes unnecessary words
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')

nltk.download('rslp')
# extract the root of word, ex:
# stemmer.stem('pode')
# 'pod'
# stemmer.stem('podem')
# 'pod'
stemmer = nltk.stem.RSLPStemmer()

# Leaves the dictionary clean
for list_of_words in broked_texts:
    valid_words = [stemmer.stem(word) for word in list_of_words if word not in stopwords and len(word) > 2]
    dictionary.update(valid_words)

#print(dictionary)
total_of_words = len(dictionary)
tuples = zip(dictionary, range(total_of_words))
indexer = {word: index for word, index in tuples}
print(f'Dictionary has: {total_of_words} words')


def vectorized_text(text, _indexer, _stemmer):
    vector = [0] * len(_indexer)
    for word in text:
        if len(word) > 0:
            root = _stemmer.stem(word)
            if root in _indexer:
                position = _indexer[root]
                vector[position] += 1

    return vector


vectors_of_text = [vectorized_text(text, indexer, stemmer) for text in broked_texts]

marks = data_frame['classificacao']

X = vectors_of_text
Y = marks.tolist()

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


# noinspection PyShadowingNames
def real_test(model, _data_of_validation, _data_of_validation_markings):
    result = model.predict(_data_of_validation)

    hits = result == _data_of_validation_markings

    total_of_hits = sum(hits)
    total_of_elements = len(_data_of_validation_markings)

    hits_rate = 100.0 * total_of_hits / total_of_elements

    msg = "Winner hit rate among real-world algorithms : {0}".format(hits_rate)
    print(msg)


results = {}

print('-' * 100)

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

print('-' * 100)
print(f"Winner: >>{winner}<<")

winner.fit(data_training, data_training_markings)

real_test(winner, data_of_validation, data_of_validation_markings)
base_hits = max(Counter(data_of_validation_markings).values())

base_hits_rate = 100.0 * base_hits / len(data_of_validation_markings)
print("Base Hits Rate: %f" % base_hits_rate)

total_of_elements = len(data_of_validation)
print("Total of tests: %d" % total_of_elements)
