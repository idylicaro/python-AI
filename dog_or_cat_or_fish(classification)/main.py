from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy

# attributes ['has_legs',long nose hairs, is_long]

dog1 = [1, 0, 1]
dog2 = [1, 0, 0]
cat1 = [1, 1, 0]
cat2 = [1, 1, 0]
fish1 = [0, 0, 1]
fish2 = [0, 0, 0]

data = [dog1, dog2, cat1, cat2, fish1, fish2]

data_marking = [1, 1, -1, -1, 2, 2]

tests = numpy.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]]).reshape(3, 3)
tests_marking = [-1, 1, 2]

model = OneVsRestClassifier(LinearSVC(random_state=0))
model.fit(data, data_marking)

result = model.predict(tests)

difference = result - tests_marking
hits = [d for d in difference if d == 0]

total_of_hits = len(hits)
total_of_elements = len(tests)

hits_rating = 100 * (total_of_hits / total_of_elements)
print(f'\n>>>The success rate of tests is:#{hits_rating}%')
